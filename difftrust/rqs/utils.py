import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

import HumanEval
import MBPP
import difftrust
from difftrust.rqs.constants import list_with_temp_appended_06_10, list_with_temp_appended_02_10, \
    list_with_temp_appended_1_10, list_with_temp_appended_06_50, list_with_temp_appended_02_50, \
    list_with_temp_appended_1_50
from difftrust.rqs.custom_classes import ExperimentConfig, ExperimentData

def make_result_dir(experiment_specific_dir):
    csv_dir = os.path.join(experiment_specific_dir, "csv")
    plots_dir = os.path.join(experiment_specific_dir, "plots")
    tex_dir = os.path.join(experiment_specific_dir, "tex")
    os.makedirs(experiment_specific_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tex_dir, exist_ok=True)
    return csv_dir, plots_dir, tex_dir

def detection_metrics(configuration, errs, diss):
    total = len(errs)
    errors = sum(1 for i in range(total) if errs[i] > 0.0)
    detected_errors = sum(1 for i in range(total) if errs[i] > 0.0 and diss[i] > 0.0)
    confident = sum(1 for i in range(total) if diss[i] == 0.0)
    confident_error = sum(1 for i in range(total) if errs[i] > 0.0 and diss[i] == 0.0)
    confident_idxs = [i for i in range(total) if diss[i] > 0.0]
    confident_errors_values = [errs[i] for i in confident_idxs]
    confident_diss_values = [diss[i] for i in confident_idxs]
    error_mean_when_confident = (
        sum(errs[i] for i in range(total) if errs[i] > 0.0 and diss[i] == 0.0) / confident
        if confident_error > 0 else 0.0
    )

    # Compute R^2 if possible
    try:
        r2 = r2_score(errs, diss)
    except ValueError:
        r2 = "undefined"  # e.g., if all values are constant
    rho, _ = spearmanr(errs, diss)
    confident_rho, _ = spearmanr(confident_errors_values, confident_diss_values)
    return {
        "Configuration": configuration,
        "Tasks Analyzed": total,
        "Raw Dis Mean": np.mean(diss),
        "Raw Err Mean": np.mean(errs),
        "Errors (Err > 0)": errors,
        "Error Rate (Errors / Total)": errors / total if total else 0,
        "Detected Errors (Err > 0 and Dis > 0)": detected_errors,
        "Detection Rate (Detected / Errors)": detected_errors / errors if errors else 0,
        "Confident (Dis = 0)": confident,
        "Confident Rate (Confident / Total)": confident / total if total else 0,
        "Confident Errors (Err > 0 and Dis = 0)": confident_error,
        "Confident Error Rate (Confident Errors / Confident)": confident_error / confident if confident else 0,
        "Error Mean When Confident": error_mean_when_confident,
        "Raw Dis Std": np.std(diss),
        "Raw Dis min": np.min(diss),
        "Raw Dis Max": np.max(diss),
        "Raw Err Std": np.std(errs),
        "Raw Err min": np.min(errs),
        "Raw Err Max": np.max(errs),
        "R² (Err vs Dis)": r2 if isinstance(r2, float) else r2,
        "Spearman Rho err vs. diss": rho,
        "Unconfident Spearman Rho err vs. diss": confident_rho,
    }


def add_rankings(experiment_specific_dir, df, csv_file="rank_df_rq2.csv"):
    """
    Add rank columns to the DataFrame based on selected numeric metrics,
    using 'first' tie-breaking method.
    """
    # Columns to rank (exclude percentage and formatted columns)
    columns_to_rank = [
        ("Raw Dis Mean", False),
        ("Raw Err Mean", True),
        ("Errors (Err > 0)", True),
        ("Error Rate (Errors / Total)", True),
        ("Detected Errors (Err > 0 and Dis > 0)", True),
        ("Detection Rate (Detected / Errors)", True),
        ("Confident (Dis = 0)", False),
        ("Confident Rate (Confident / Total)", False),
        ("Confident Errors (Err > 0 and Dis = 0)", True),
        ("Confident Error Rate (Confident Errors / Confident)", True),
        ("Error Mean When Confident", True),
        ("Spearman Rho err vs. diss", True),
    ]

    rank_df = pd.DataFrame(index=df.index)
    rank_df["Configuration"] = df["Configuration"]

    for col, ascending_or_not in columns_to_rank:
        if col in df.columns:
            rank_df[f"{col} Rank"] = df[col].rank(method="first", ascending=ascending_or_not)
    if csv_file != "":
        csv_path = os.path.join(experiment_specific_dir, "csv")
        os.makedirs(csv_path, exist_ok=True)
        csv_file_full_path = os.path.join(csv_path, csv_file)
        rank_df.to_csv(csv_file_full_path)
    return rank_df

def get_all_configs(main_exp_top_level_dir):
    all_configs = {}
    all_configs['06_10'] = load_configurations_data(list_with_temp_appended_06_10,
                                                    experiment_dir=main_exp_top_level_dir)
    all_configs['02_10'] = load_configurations_data(list_with_temp_appended_02_10,
                                                    experiment_dir=main_exp_top_level_dir)
    all_configs['1_10'] = load_configurations_data(list_with_temp_appended_1_10, experiment_dir=main_exp_top_level_dir)
    all_configs['06_50'] = load_configurations_data(list_with_temp_appended_06_50,
                                                    experiment_dir=main_exp_top_level_dir)
    all_configs['02_50'] = load_configurations_data(list_with_temp_appended_02_50,
                                                    experiment_dir=main_exp_top_level_dir)
    all_configs['1_50'] = load_configurations_data(list_with_temp_appended_1_50, experiment_dir=main_exp_top_level_dir)

    return all_configs

def load_configurations_data(model_list, experiment_dir=".experiments", log_file_name="experiment-pointwise"):
    all_configs = {
        model_name: load_model_data(model_name, experiment_dir, log_file_name)
        for model_name in model_list
    }
    return all_configs

def load_model_data(config_="gpt_4o", experiment_dir=".experiments", log_file_name="experiment-pointwise"):
    ctxt = difftrust.experiment.ExperimentCtxt(f"{experiment_dir}/{config_}")
    logs = ctxt.get_logs(log_file_name)

    filtered_logs = [log for log in logs if "Failed" not in log]

    raw_dis = [log.get("Dis", 0.0) for log in filtered_logs]
    raw_err = [log.get("Err", 0.0) for log in filtered_logs]
    ref_dis = [log.get("RefinedDis", 0.0) for log in filtered_logs]
    ref_err = [log.get("RefinedErr", 0.0) for log in filtered_logs]
    names = [log.get("name", "") for log in filtered_logs]

    total = len(logs)
    config_obj = ExperimentConfig.from_dict(ctxt.config)
    data_obj = ExperimentData(logs=filtered_logs, raw_dis=raw_dis, raw_err=raw_err,
                              ref_dis=ref_dis, ref_err=ref_err, names=names, total=total)
    return config_obj, data_obj

def run_parameterized_experiment(instance, ExperimentRunner, llm_name, temp, num_candidates,
                                 num_samples, timeout, suffix, experiment_dir):
    print(f"Model is {llm_name}")
    ctxt = difftrust.experiment.ExperimentCtxt(f"{experiment_dir}/{llm_name}_temp{temp}")
    ctxt.set_config("llm", llm_name)
    ctxt.set_config("nb_candidate", num_candidates)
    ctxt.set_config("nb_sample", num_samples)
    ctxt.set_config("temperature", temp)
    ctxt.set_config("timeout", timeout)
    print(f"Model is {llm_name} temp is {ctxt.get_config("temperature")}")
    dataset = instance.Dataset("dataset-v1")
    # dataset.make()
    dataset.load()
    # new_dataset = copy(dataset)
    # new_instances = regenerate(dataset.instances)
    # new_dataset.instances = new_instances

    runner = ExperimentRunner(ctxt, dataset)

    runner.run("pointwise", call_llm_api=False, suffix=suffix, slow_mode=False)

def filter_by_config(configs, **filters):
    """
    Filter a dictionary of LLMConfig instances using exact and numeric comparisons.

    Supported operations:
      - __lt, __lte, __gt, __gte, __ne, __eq
      - default key (no __) is treated as equality

    Example:
        filter_by_config(configs, llm="gpt-4", nb_sample__gte=1000, temperature__eq=0.6)
    """

    def passes(llm_data):
        config = llm_data[0]
        for key, value in filters.items():
            if '__' in key:
                attr, op = key.split('__', 1)
                actual = getattr(config, attr, None)
                if actual is None:
                    return False
                if op == 'lt' and not (actual < value): return False
                if op == 'lte' and not (actual <= value): return False
                if op == 'gt' and not (actual > value): return False
                if op == 'gte' and not (actual >= value): return False
                if op == 'ne' and not (actual != value): return False
                if op == 'eq' and not (actual == value): return False
            else:
                if getattr(config, key, None) != value:
                    return False
        return True

    return {k: v for k, v in configs.items() if passes(v)}

def save_pkl(obj, filepath):
    """Save a Python object to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(filepath):
    """Load a Python object from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)



def get_instance_and_runner(benchmark_name):
    # Monkey patching hack for the win
    if benchmark_name == "MBPP":
        sys.modules['instance'] = MBPP.instance
        sys.modules['ExperimentRunner'] = MBPP.run.ExperimentRunner
        return MBPP.instance, MBPP.run.ExperimentRunner
    else:
        sys.modules['instance'] = HumanEval.instance
        sys.modules['ExperimentRunner'] = HumanEval.run.ExperimentRunner

        return HumanEval.instance, HumanEval.run.ExperimentRunner

