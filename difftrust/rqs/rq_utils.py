import shutil
from datetime import datetime
from constants import *
from difftrust.rqs.ablation import ablation_helper
from difftrust.rqs.custom_classes import ExperimentData
from difftrust.rqs.latex import generate_small_table_of_comparison_latex, generate_full_table_of_comparison_latex
from difftrust.rqs.utils import add_rankings, get_all_configs, filter_by_config, load_pkl, save_pkl, make_result_dir
from plots import *


def RQ1(results_dir, trial_data_06_sample_1000, benchmark_name, threshold):
    gpt_4o_candidates = trial_data_06_sample_1000['gpt_4o_temp0.6'][threshold][1]

    RQ1_1(results_dir, gpt_4o_candidates, benchmark_name, threshold)


def RQ1_1(results_dir, model: ExperimentData, benchmark_name, threshold, label="GPT-4o", name="GPT-4o",
          cmap="viridis", ):
    raw_dis, raw_err = model.raw_dis, model.raw_err
    save_grid_plot(results_dir, raw_dis, raw_err, f"rq1-grid-plot-{label}-{threshold} functions -{benchmark_name}",
                   'mako_r')
    save_log2_scatter_plot(results_dir, raw_dis, raw_err, alternate_label=f"{benchmark_name}",
                           file_name=f"rq1-scatter-plot-{label}-{threshold}-functions-{benchmark_name}",
                           title=f"{label} -{threshold} functions - Error vs. Incoherence Scatter Plot (log₂ scale)")


def RQ2_1(experiment_specific_dir, df_1, benchmark_name):
    rank_df = add_rankings(experiment_specific_dir, df_1, csv_file=f"rank_df_rq2_{benchmark_name}.csv")
    return rank_df


def RQ2(experiment_specific_dir, rank_df, benchmark_name, threshold):

    plot_rank_scatter(rank_df, "Error Rate (Errors / Total) Rank", "Confident Rate (Confident / Total) Rank",
                      "Ranking by the Number of Functions Deemed Correct via Oracle",
                      " Ranking by the Number of Functions Deemed Correct via Incoherence", experiment_specific_dir,
                      "plots", benchmark_name, file_stub=f"rq2-{threshold}-{benchmark_name}")
    print("RQ2 Done.")


def temp_incoherence_helper(prefix, ablation_df, x_label, results_dir="results",
                            image_dir="metric_plot_over_thresholds"):
    """
    Plot 'Error Rate (Errors / Total)' and 'Confident Rate (Confident / Total)' as integer percentages
    over thresholds for each model, including a mean line. Saves one plot per metric.
    """
    os.makedirs(os.path.join(results_dir, image_dir), exist_ok=True)

    metrics_to_plot = ["Error Rate (Errors / Total)", "Confident Rate (Confident / Total)"]

    # Create a copy to safely modify
    ablation_df_percent = ablation_df.copy()

    for col in metrics_to_plot:
        if ablation_df_percent[col].dtype == object:
            # Strip '%' if present, convert to float
            ablation_df_percent[col] = (
                ablation_df_percent[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .astype(float)
            )
        # Scale only if values look like proportions (e.g., < 1.0)
        if (ablation_df_percent[col] <= 1.0).all():
            ablation_df_percent[col] *= 100

        ablation_df_percent[col] = ablation_df_percent[col].round().astype(int)

    # Melt original data
    melted_df = ablation_df_percent.melt(
        id_vars=["model", "threshold"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Value"
    )

    # Compute mean across models at each threshold
    mean_df = (
        ablation_df_percent
        .groupby("threshold")[metrics_to_plot]
        .mean(numeric_only=True)
        .round()
        .astype(int)
        .reset_index()
        .melt(id_vars="threshold", var_name="Metric", value_name="Value")
    )
    mean_df["model"] = "mean"

    # Combine original + mean
    full_plot_df = pd.concat([melted_df, mean_df], ignore_index=True)

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))

        plot_df = full_plot_df[full_plot_df["Metric"] == metric]

        sns.lineplot(
            data=plot_df,
            x="threshold",
            y="Value",
            hue="model",
            marker="o"
        )

        plt.title(f"{metric} vs {x_label}")
        plt.xlabel(x_label)
        plt.ylabel("Percentage (%)")
        plt.grid(True)
        plt.tight_layout()

        file_name = f"{prefix}_{metric}".lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        plot_path = os.path.join(results_dir, image_dir, f"{file_name}_over_thresholds.png")

        plt.savefig(plot_path, dpi=300)
        plt.show()
        plt.close()

    return full_plot_df


def RQ3(exp_top_level_dir, benchmark_name, experiment_specific_dir, all_configs, default_temp=0.6,
        default_sample_size=1000, run_rq3_1_=True, run_rq3_2_=True, run_rq3_3_=True):
    rq3_1_ = {}
    if run_rq3_1_:
        for config__ in ['06_50']:
            temp_ablation_df_rq3_1, temp_rank_ablation_df_rq3_1, temp_ablation_data_rq3_1 = ablation_helper(
                exp_top_level_dir, benchmark_name,
                experiment_specific_dir,
                all_configs[config__],
                default_input_sample=default_sample_size,
                project_temp=default_temp,
                ablation_models=models_with_fifty_entries,
                rq_label=f"rq3_1_{config__}")
            rq3_1_[config__] = [temp_ablation_df_rq3_1, temp_rank_ablation_df_rq3_1, temp_ablation_data_rq3_1]

            plot_rank_scatter(temp_rank_ablation_df_rq3_1, "Error Rate (Errors / Total) Rank",
                              "Confident Rate (Confident / Total) Rank",
                              "Ranking by the Number of Functions Deemed Correct via Oracle",
                              " Ranking by the Number of Functions Deemed Correct via Incoherence",
                              experiment_specific_dir,
                              "plots", file_stub=f"candidate_ablation_{config__}", alternate_text=None)

            temp_incoherence_helper(f"Candidate Function Ablation {benchmark_name}", temp_ablation_df_rq3_1,
                                    "Candidate Function", results_dir=experiment_specific_dir, image_dir="plots")

    rq3_2_ = {}
    if run_rq3_2_:
        ablation_df_rq3_2, rank_ablation_df_rq3_2, temp_ablation_data_rq3_2 = ablation_helper(exp_top_level_dir,
                                                                                              benchmark_name,
                                                                                              experiment_specific_dir,
                                                                                              all_configs['06_10'],
                                                                                              mode_="sample",
                                                                                              experiment_specific_dir=".experiments-ablation-samples",
                                                                                              rq_label="rq3_2",
                                                                                              ablation_models=models_with_ten_entries)
        rq3_2_['06_10'] = [ablation_df_rq3_2, rank_ablation_df_rq3_2, temp_ablation_data_rq3_2]

        plot_rank_scatter(rank_ablation_df_rq3_2, "Error Rate (Errors / Total) Rank",
                          "Confident Rate (Confident / Total) Rank",
                          "Ranking by the Number of Functions Deemed Correct via Oracle",
                          " Ranking by the Number of Functions Deemed Correct via Incoherence", experiment_specific_dir,
                          "plots", file_stub=f"sample_ablation_rq3_2_{benchmark_name}", alternate_text=None)
        temp_incoherence_helper(f"Sample Ablation {benchmark_name}", ablation_df_rq3_2, "Test Case Samples",
                                results_dir=experiment_specific_dir, image_dir="plots")

    rq3_3_ = {}
    if run_rq3_3_:
        filtered_06_10 = {k: v for k, v in all_configs['06_10'].items() if k.startswith(tuple(all_temp_models))}
        combined_configs_for_temp = filtered_06_10
        combined_configs_for_temp.update(all_configs['02_10'])
        combined_configs_for_temp.update(all_configs['1_10'])
        ablation_df_rq3_3, rank_ablation_df_rq3_3, temp_ablation_data_rq3_2 = ablation_helper(exp_top_level_dir,
                                                                                              benchmark_name,
                                                                                              experiment_specific_dir,
                                                                                              combined_configs_for_temp,
                                                                                              mode_="temp",
                                                                                              experiment_specific_dir=".experiments-ablation-temps",
                                                                                              rq_label="rq3_3")
        rq3_3_["temp"] = [ablation_df_rq3_3, rank_ablation_df_rq3_3, temp_ablation_data_rq3_2]
        plot_rank_scatter(rank_ablation_df_rq3_3, "Error Rate (Errors / Total) Rank",
                          "Confident Rate (Confident / Total) Rank",
                          "Ranking by the Number of Functions Deemed Correct via Oracle",
                          " Ranking by the Number of Functions Deemed Correct via Incoherence", experiment_specific_dir,
                          "plots", file_stub=f"temp_ablation_rq3_3_{benchmark_name}", alternate_text=None)
        melted__ = temp_incoherence_helper(f"Temperature Ablation {benchmark_name}", ablation_df_rq3_3, "Temperature",
                                           results_dir=experiment_specific_dir, image_dir="plots")
    print("All Done!")
    return rq3_1_, rq3_2_, rq3_3_


def copy_to_latest(experiment_specific_dir: str, top_level_folder: str, latest_dir: str):
    latest_path = os.path.join(top_level_folder, latest_dir)
    shutil.copytree(experiment_specific_dir, latest_path, dirs_exist_ok=True)
    log_file = os.path.join(latest_path, "log.log")
    with open(log_file, "w") as l:
        l.write(f"Orginal folder was experiment_specific_dir")
    print(f"Copied {experiment_specific_dir} to {latest_path}")


def pregenerate_results_for_config(exp_top_level_dir, dataset, results_dir, all_configs,
                                   num_candidates=10,
                                   default_input_sample=1000, experiment_specific_dir=".experiments",
                                   project_temp=0.6,
                                   rq_label="rq1_06_10_config", mode_="candidate",
                                   experiment_type="experiment-pointwise", hard_coded_threshold=None):
    # Recompute the values for 10 candidates if there are 50 candidates in the experiment
    # as if we are doing ablation for 10 candidates.
    ablation_modelz = [i for i in models_with_ten_entries] if hard_coded_threshold[0] == 10 else [i for i in
                                                                                                  models_with_fifty_entries]
    df, rank_df, all_ablation_data = ablation_helper(exp_top_level_dir, dataset, results_dir, all_configs,
                                                     default_candidates=num_candidates,
                                                     default_input_sample=default_input_sample,
                                                     experiment_specific_dir=experiment_specific_dir,
                                                     project_temp=project_temp,
                                                     rq_label=rq_label, mode_=mode_, experiment_type=experiment_type,
                                                     ablation_models=ablation_modelz,
                                                     hard_coded_threshold=hard_coded_threshold)
    return df, rank_df, all_ablation_data


def rqs_1_and_2(main_ranges):
    results = {}
    for threshold in main_ranges:
        temp_06_sample_1000 = filter_by_config(all_configs[f'06_{threshold}'], nb_sample__eq=1000, temperature__eq=0.6)
        df_06_sample_1000, rank_data_06_sample_1000, trial_data_06_sample_1000 = pregenerate_results_for_config(
            exp_top_level_dir, benchmark_name, experiment_specific_dir,
            temp_06_sample_1000, num_candidates=threshold,
            default_input_sample=1000, experiment_specific_dir=".experiments",
            project_temp=0.6,
            rq_label=f"rq1_06_{threshold}_config", mode_="candidate",
            experiment_type="experiment-pointwise", hard_coded_threshold=[threshold])
        # Create a directory for output plots
        RQ1(experiment_specific_dir, trial_data_06_sample_1000, benchmark_name, threshold)
        RQ2(experiment_specific_dir, rank_data_06_sample_1000, benchmark_name, threshold)
        results[threshold] = [df_06_sample_1000, rank_data_06_sample_1000, trial_data_06_sample_1000]
    return results


if __name__ == "__main__":
    load_1_and_2 = True
    skip_1_and_2 = True
    skip_3 = True
    results = {}
    experiment_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    top_level_folder = "aggregate-plots"
    experiment_specific_dir = os.path.join(top_level_folder, experiment_time)
    results_pkl = f"{top_level_folder}/0_latest/results.pkl"
    benchmarks = ["MBPP", "HumanEval"]
    csv_dir, plots_dir, tex_dir = make_result_dir(experiment_specific_dir)
    if load_1_and_2:
        results = load_pkl(results_pkl)

    for benchmark_name in benchmarks:
        print(f"Currently working on benchmark {benchmark_name}")
        exp_top_level_dir = os.path.abspath(os.path.join(__file__, f"../../../{benchmark_name}"))
        main_exp_top_level_dir = os.path.join(exp_top_level_dir, ".experiments")
        all_configs = get_all_configs(main_exp_top_level_dir)
        if not skip_1_and_2:
            results[benchmark_name] = rqs_1_and_2(main_ranges)

        if not skip_3:
            results[f"{benchmark_name}_rq3"] = RQ3(exp_top_level_dir, benchmark_name, experiment_specific_dir,
                                                   all_configs)

        save_pkl(results, results_pkl)
    generate_small_table_of_comparison_latex(results, 10, f"{tex_dir}/small_table_of_comparison_latex.tex",
                                             models_for_comparison_table)
    generate_full_table_of_comparison_latex(results, 10, f"{tex_dir}/full_table_of_comparison_latex.tex",
                                            [i for i in models_with_ten_entries])
    # Merge the results for each dataset
    copy_to_latest(experiment_specific_dir, top_level_folder, "0_latest")
    print("Done.")
