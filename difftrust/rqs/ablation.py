import os

import pandas as pd

from difftrust.rqs.constants import *
from difftrust.rqs.utils import load_configurations_data, detection_metrics, run_parameterized_experiment, \
    load_pkl, load_model_data, save_pkl, add_rankings, get_instance_and_runner


def get_ablation_range(mode_, hard_coded_threshold=None):
    if hard_coded_threshold:
        return hard_coded_threshold
    if mode_ == "candidate":
        return ablation_candidate_range
    elif mode_ == "sample":
        return ablation_samples_range
    elif "temp":
        return ablation_temps


def ablation_first_pass(inputss):
    benchmark_name, ablation_range, mode_, project_temp, model_, top_level_results_dir, rq_label, experiment_type, filtered_projects_for_threshold, default_candidates, default_input_sample, ablation_data = inputss
    instance, ExperimentRunner = get_instance_and_runner(benchmark_name)

    for threshold_ in ablation_range:
        project_temp, model_temp_key, model_results_dir, ablation_pkls_path, ablation_pkls_path_file_pkl_path, ablation_suffix = (
            get_model_threshold_keys(
                threshold_, mode_, project_temp, model_, top_level_results_dir, rq_label))

        print(f"Working on model: {model_} threshold: {threshold_} mode: {mode_}")
        #  if the key exists and override doesn't exist for the key, skip.
        if model_temp_key in ablation_data and threshold_ in ablation_data[
            model_temp_key] and not f"{model_temp_key}_{threshold_}" in overrides:
            continue
        #  otherwise, generate the data if json doesn't exist
        llmconfig, _ = filtered_projects_for_threshold[model_temp_key]
        run_parameterized_experiment(instance, ExperimentRunner, llmconfig.llm, llmconfig.temperature,
                                     threshold_ if mode_ == "candidate" else default_candidates,
                                     threshold_ if mode_ == "sample" else default_input_sample, llmconfig.timeout,
                                     ablation_suffix,
                                     top_level_results_dir)
        print(f"Finished with {model_temp_key}")


def get_model_threshold_keys(threshold_, mode_, project_temp, model_, top_level_results_dir, rq_label):
    project_temp = threshold_ if mode_ == "temp" else project_temp
    model_temp_key = f"{model_}_temp{project_temp}"
    model_results_dir = os.path.join(top_level_results_dir, model_temp_key)
    ablation_pkls_path = os.path.join(model_results_dir, "pkls")
    ablation_pkls_path_file_pkl_path = os.path.join(ablation_pkls_path, f"{rq_label}.pkl")
    ablation_suffix = f"{threshold_}_{mode_}"
    return project_temp, model_temp_key, model_results_dir, ablation_pkls_path, ablation_pkls_path_file_pkl_path, ablation_suffix


def ablation_helper(exp_top_level_dir, benchmark_name, results_dir, all_configs, default_candidates=10,
                    default_input_sample=1000, experiment_specific_dir=".experiments-ablation", project_temp=0.6,
                    rq_label="rq3_1", mode_="candidate", experiment_type="experiment-pointwise",
                    ablation_models=ablation_models,
                    alternate_text=None, hard_coded_threshold=None):
    alternate_text = rq_label if alternate_text is None else alternate_text
    top_level_results_dir = os.path.join(exp_top_level_dir, experiment_specific_dir)

    ablation_range = get_ablation_range(mode_, hard_coded_threshold)

    all_ablation_data = {}

    # load cached data here for ablation data here so all processes can have read access

    for model_ in ablation_models:
        for threshold_ in ablation_range:
            project_temp, model_temp_key, model_results_dir, ablation_pkls_path, ablation_pkls_path_file_pkl_path, ablation_suffix = get_model_threshold_keys(
                threshold_, mode_, project_temp, model_, top_level_results_dir, rq_label)

            if os.path.exists(ablation_pkls_path_file_pkl_path):
                all_ablation_data.update(load_pkl(ablation_pkls_path_file_pkl_path))

    # first pass. Data is stored in jsons but if they have reached and stored correctly in second pass, they will be ignored
    # for model_ in ablation_models:

    from multiprocess import Pool
    inputss = [(benchmark_name, ablation_range, mode_, project_temp, model_,
                top_level_results_dir, rq_label, experiment_type, all_configs,
                default_candidates, default_input_sample, all_ablation_data) for model_ in ablation_models]
    with Pool(processes=len(ablation_models)) as pool:
        pool.map(ablation_first_pass, inputss)

    # second pass. Back up into pkls for future
    for model_ in ablation_models:
        ablation_second_pass(ablation_range, all_ablation_data, experiment_type, mode_, model_,
                             project_temp, rq_label, top_level_results_dir)

    # third pass.
    ablation_df = pd.DataFrame()
    # calculate ablation stuff and store
    for model_ in ablation_models:
        for threshold_ in ablation_range:
            project_temp, model_temp_key, model_results_dir, ablation_pkls_path, ablation_pkls_path_file_pkl_path, ablation_suffix = get_model_threshold_keys(
                threshold_, mode_, project_temp, model_, top_level_results_dir, rq_label)

            temp_data_ = all_ablation_data[model_temp_key][threshold_]
            ablation_temp_dict = detection_metrics(model_temp_key, temp_data_[1].raw_err, temp_data_[1].raw_dis)
            ablation_temp_dict['no_candidates'] = threshold_
            ablation_temp_dict["Configuration"] = f"{model_temp_key}_{ablation_suffix}"
            ablation_temp_dict["threshold"] = threshold_
            ablation_temp_dict["model"] = model_
            temp_df = pd.DataFrame([ablation_temp_dict])
            temp_df['Detection Rate (Detected / Errors)'] = pd.to_numeric(
                temp_df['Detection Rate (Detected / Errors)'], errors='coerce'
            )
            temp_df['Error Mean When Confident'] = pd.to_numeric(
                temp_df['Error Mean When Confident'], errors='coerce'
            )
            ablation_df = pd.concat([ablation_df, temp_df], ignore_index=True)

    rank_ablation_df = add_rankings(results_dir, ablation_df[ablation_df["threshold"] == ablation_range[-1]],
                                    csv_file=f"rank_df_rq2_{benchmark_name}.csv")

    rank_ablation_df['model'] = rank_ablation_df['Configuration'].map(ablation_df.set_index('Configuration')['model'])
    rank_ablation_df['threshold'] = rank_ablation_df['Configuration'].map(
        ablation_df.set_index('Configuration')['threshold'])
    ablation_df.to_csv(f"{results_dir}/csv/ablation_df_{rq_label}_{mode_}_{benchmark_name}.csv")
    rank_ablation_df.to_csv(f"{results_dir}/csv/rank_ablation_df_{rq_label}_{mode_}_{benchmark_name}.csv")

    return ablation_df, rank_ablation_df, all_ablation_data


def ablation_second_pass(ablation_range, all_ablation_data, experiment_type, mode_, model_, project_temp, rq_label,
                         top_level_results_dir):
    ablation_data = {}
    for threshold_ in ablation_range:
        project_temp, model_temp_key, model_results_dir, ablation_pkls_path, ablation_pkls_path_file_pkl_path, ablation_suffix = get_model_threshold_keys(
            threshold_, mode_, project_temp, model_, top_level_results_dir, rq_label)
        # try pkl
        if os.path.exists(ablation_pkls_path_file_pkl_path) and model_temp_key in ablation_data and threshold_ in \
                ablation_data[model_temp_key]:
            ablation_data = load_pkl(ablation_pkls_path_file_pkl_path)
            all_ablation_data[model_temp_key] = ablation_data[model_temp_key]
        else:
            if model_temp_key not in ablation_data:
                ablation_data[model_temp_key] = {}
            if model_temp_key not in all_ablation_data:
                all_ablation_data[model_temp_key] = {}
            # import from json
            json_file_name = f"{experiment_type}_{ablation_suffix}.json"
            model_data_from_json = {
                threshold_: load_model_data(model_temp_key, top_level_results_dir, json_file_name[:-5])}
            ablation_data[model_temp_key].update(model_data_from_json)
            all_ablation_data[model_temp_key].update(model_data_from_json)
            #  store pkl
            save_pkl(ablation_data, ablation_pkls_path_file_pkl_path)
    print("Finished second pass.")


def get_ablation_temp_combos():
    model_groups_listss = [model for model in ablation_models]
    list_with_temp_appended_02 = append_temp_to_name(model_groups_listss, .2)
    list_with_temp_appended_1 = append_temp_to_name(model_groups_listss, 1)
    new_configs = load_configurations_data(list_with_temp_appended_02, ".experiments-ablation-temps")
    new_configs.update(load_configurations_data(list_with_temp_appended_1, ".experiments-ablation-temps"))
    return new_configs
