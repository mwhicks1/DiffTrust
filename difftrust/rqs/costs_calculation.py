# Define pricing for each model based on public information and reasonable assumptions (as of mid-2025).
# Prices are per 1,000 tokens in USD.
import os
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
import tiktoken
from tqdm import tqdm

from difftrust.core.function import Function
from difftrust.llm.claude import claude_sonnet_4, claude_opus_4
from difftrust.rqs.constants import models_with_ten_entries, all_temp_models, normalize_model_name, model_pricing, \
    model_token_divisors, config_to_ablation_csv_mapping, candidate_thresholds, samples_thresholds, temp_thresholds
from difftrust.rqs.latex import generate_latex_table_ablation


# Function to estimate cost
def estimate_cost(prompt_tokens: int, completion_tokens: int):
    cost_estimates = []
    for model, (prompt_price, completion_price) in model_pricing.items():
        prompt_cost = (prompt_tokens / 1000000) * prompt_price
        completion_cost = (completion_tokens / 1000000) * completion_price
        total_cost = prompt_cost + completion_cost
        cost_estimates.append({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_cost_usd": round(total_cost, 6)
        })
    return pd.DataFrame(cost_estimates).sort_values(by="total_cost_usd")


def get_token_count(model: str, prompt: str, completion: str) -> tuple[int, int]:
    if model.startswith("gpt"):
        model_ = normalize_model_name(model)
        # Use OpenAI's token counting via tiktoken
        prompt_tokens = count_openai_tokens(model_, [{"role": "user", "content": prompt}])
        completion_tokens = len(tiktoken.encoding_for_model(model_).encode(completion))
        return (prompt_tokens, completion_tokens)

    elif model.startswith("claude"):
        client = claude_sonnet_4 if "sonnet" in model else claude_opus_4
        result_prompt = client.get_token_count(
            system_prompt="You are world class python programmer",
            messages=[{
                "role": "user",
                "content": f"{prompt}"}
            ]).input_tokens
        result_completion = client.get_token_count(
            system_prompt="You are world class python programmer",
            messages=[{
                "role": "assistant",
                "content": f"{completion}"}
            ]).input_tokens
        return (result_prompt, result_completion)

    else:

        return (
            int(max(1, int(len(prompt) / model_token_divisors[model][0]))),
            int(max(1, int(len(completion) / model_token_divisors[model][1])))
        )


def estimate_cost_from_text(model: str, prompt: str, completion: str):
    prompt_tokens, completion_tokens = get_token_count(model, prompt, completion)
    return estimate_cost(prompt_tokens, completion_tokens)


def count_openai_tokens(model_: str, messages: list[dict]) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    if model_.startswith("gpt-3.5") or model_.startswith("gpt-4"):
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"Token counting not supported for model: {model_}")

    num_tokens = 0
    for msg in messages:
        num_tokens += tokens_per_message
        for key, value in msg.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name

    num_tokens += 3  # primed reply
    return num_tokens


def get_question_text(spec):
    return (
        f"Can you write a function with the following specification:\n\n"
        f"{spec}\n\n"
        f"Please write the full code in one go like this :\n\n"
        f"```python\n"
        f"# imports ...\n"
        f"# Auxiliary functions and classes ...\n"
        f"def {spec.name}{spec.signature}:\n"
        f"    # {spec.name} code ...\n"
        f"# End of python code\n"
        f"```\n\n"
        f"Please ensure the function adheres strictly to the provided specifications.\n"
    )


def get_single_candidate_cost(func_: Function, model):
    prompt = get_question_text(func_.spec)
    completion = func_.code
    prompt_tokens, completion_tokens = get_token_count(model, prompt, completion)
    cost = estimate_cost(prompt_tokens, completion_tokens).query(f"model == '{model}'").iloc[0]
    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_cost_usd": cost["total_cost_usd"]
    }


def get_configuration_cost(root, models: list, dataset_mapping: dict, experiment_suffix: str,
                           n_first: int = 10) -> pd.DataFrame:
    cost_records = []

    for dataset, dataset_rel_path in dataset_mapping.items():
        print(f"Working on {dataset} costs")
        dataset_dir = root / dataset_rel_path
        if not dataset_dir.exists():
            print(f"[WARN] Dataset directory missing: {dataset_dir}")
            continue

        for model in models:

            print(f"-- Working on {model}")
            model_dir = dataset_dir / f"{model}_{experiment_suffix}"
            if not model_dir.exists():
                print(f"[WARN] Skipping missing model dir: {model_dir}")
                continue

            config_dir = model_dir / "cache"
            if not config_dir.exists():
                print(f"[WARN] Missing cache directory: {config_dir}")
                continue

            for pickle_file in tqdm(config_dir.iterdir()):
                if "refined" in pickle_file.name:
                    continue
                try:
                    with open(pickle_file, "rb") as f:
                        candidates = pickle.load(f)
                except Exception as e:
                    print(f"[ERROR] Reading {pickle_file}: {e}")
                    continue

                for i, func_ in enumerate(candidates):
                    try:
                        cost = get_single_candidate_cost(func_, model)
                        cost.update({
                            "dataset": dataset,
                            "dataset_rel_path": str(dataset_rel_path),
                            "config": experiment_suffix,
                            "candidate_file": pickle_file.name,
                            "candidate_index": i,
                        })
                        cost_records.append(cost)
                    except Exception as e:
                        print(f"[ERROR] Processing candidate #{i} in {pickle_file}: {e}")
    df = pd.DataFrame(cost_records)
    return df


def compute_per_candidate_pricing(dataset_mapping, dfs, prices_csv_path):
    price_mapping = {}
    for dataset, _ in dataset_mapping.items():
        current_data_set = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        # Structure: config_ -> (dataset, model) -> candidate_file -> idx -> cost

        for config_, df in dfs.items():
            df_filtered = df[df["dataset"] == dataset]

            if df_filtered.empty:
                continue

            grouped = df_filtered.groupby(["dataset", "model", "candidate_file"])

            for (ds, model, cand_file), group in grouped:
                for idx in group["candidate_index"].unique():
                    cost = group[group["candidate_index"] == idx]["total_cost_usd"].sum()
                    current_data_set[config_][(ds, model)][cand_file][idx] = cost

        price_mapping[dataset] = current_data_set
    # Flatten and compute stats
    print("last phase")
    flat_rows = []
    for dataset, dataset_data in price_mapping.items():
        for config_, combo_dict in dataset_data.items():
            for (ds, model), file_dict in combo_dict.items():
                for cand_file, idx_costs in file_dict.items():
                    sorted_indices = sorted(idx_costs.keys())
                    cumulative = 0.0
                    total_sum = sum(idx_costs.values())
                    count = 1
                    for idx in sorted_indices:
                        cost = idx_costs[idx]
                        cumulative += cost
                        mean_cost = cumulative / count if count > 0 else 0.0

                        flat_rows.append({
                            "dataset": dataset,
                            "config": config_,
                            "model": model,
                            "candidate_file": cand_file,
                            "candidate_index": idx,
                            "total_cost_usd": cost,
                            "cumulative_cost_usd": cumulative,
                            "mean_price_per_candidate_file": mean_cost
                        })
                        count += 1
    df_prices = pd.DataFrame(flat_rows)
    df_prices.to_csv(prices_csv_path, index=False)
    print("price info saved done.")
    return df_prices


def compute_per_config_prices(prices_df, stats_csv):
    # Group by (dataset, config, model) and compute:
    # - mean of mean_price_per_candidate_file (across candidate files)
    # - sum of cumulative cost (across candidate files)

    grouped = prices_df.groupby(['dataset', 'config', 'model', 'candidate_index'])

    stats = grouped.agg({
        'mean_price_per_candidate_file': 'mean',
        'cumulative_cost_usd': 'sum'
    }).reset_index()

    stats.rename(columns={
        'mean_price_per_candidate_file': 'config_mean_per_query_cost',
        'cumulative_cost_usd': 'config_total_cost'
    }, inplace=True)
    stats["no_candidates"] = stats['candidate_index'] + 1
    stats.to_csv(stats_csv, index=False)
    return stats


def merge_campaign_and_pricing_data(df_stats, csv_dir, config_to_ablation_df_mapping, models_and_their_labels):
    shared_columns_across_all = ['dataset', 'model', 'mean error', 'mean incoherence', 'spearman correlation',
                                 'detection rate', 'total cost', 'avg. cost']

    columns = [
        "Raw Err Mean",
        "Raw Dis Mean",
        "Spearman Rho err vs. diss",
        "Detection Rate (Detected / Errors)",
        "Error Mean When Confident"
    ]

    candidates_df = concat_dfs_of_same_abaltion(config_to_ablation_df_mapping, "candidate")
    candidates_df = filter_by_models(models_and_their_labels.keys(), candidates_df)
    merged_candidates_df = pd.merge(candidates_df, df_stats, how='inner', left_on=['dataset', 'model', 'no_candidates'],
                                    right_on=['dataset', 'model', 'no_candidates'])

    # \item as the number of sampled test inputs per function increases.
    samples_df = concat_dfs_of_same_abaltion(config_to_ablation_df_mapping, "sample")
    samples_df = filter_by_models(models_and_their_labels.keys(), samples_df)
    # hard coding these
    samples_df['no_candidates'] = 10
    samples_df['temp'] = 0.6

    # Rename temp to config beforehand to match df_stats (optional)
    samples_df = samples_df.rename(columns={'temp': 'config'})

    merged_samples_df = pd.merge(
        samples_df,
        df_stats,
        how='inner',
        on=['dataset', 'model', 'no_candidates', 'config']
    )

    #  \item as the temperature increases.
    temp_df = concat_dfs_of_same_abaltion(config_to_ablation_df_mapping, "temp")
    temp_df = filter_by_models(models_and_their_labels.keys(), temp_df)
    # hard coding these
    temp_df['no_candidates'] = 10
    merged_temp_df = pd.merge(temp_df, df_stats, how='inner',
                              left_on=['dataset', 'model', 'no_candidates', 'threshold'],
                              right_on=['dataset', 'model', 'no_candidates', 'config'])

    return {'candidate': merged_candidates_df, 'sample': merged_samples_df, 'temp': merged_temp_df}


def concat_dfs_of_same_abaltion(config_to_ablation_df_mapping, ablation):
    dfs_with_dataset_col = [
        df.assign(dataset=dataset_key)
        for dataset_key, df in config_to_ablation_df_mapping[ablation].items()
    ]
    return pd.concat(dfs_with_dataset_col, ignore_index=True)


def filter_by_models(models, df):
    return df[df['model'].isin(models)]


def get_per_dataset_means(df):
    numeric_cols = df.select_dtypes(include='number').columns
    non_index_numeric_cols = [col for col in numeric_cols if col != 'dataset']
    return df.groupby("dataset")[non_index_numeric_cols].mean().reset_index()


def load_ablation_csvs(config_to_ablation_csv_mapping, root_dir):
    loaded_mapping = {}

    for config, dataset_dict in config_to_ablation_csv_mapping.items():
        loaded_mapping[config] = {}

        for dataset, csv_name in dataset_dict.items():
            csv_path = os.path.join(root_dir, csv_name)
            try:
                loaded_mapping[config][dataset] = pd.read_csv(csv_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"CSV not found: {csv_path}")

    return loaded_mapping


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    results = root / "difftrust/rqs/aggregate-plots/0_latest/"
    default_csv_dir = results / "csv"
    default_latex_dir = results / "tex"
    dataset_mapping = {"MBPP": "MBPP/.experiments", "HumanEval": "HumanEval/.experiments"}
    loaded_config_to_ablation_csv_mapping = load_ablation_csvs(config_to_ablation_csv_mapping, default_csv_dir)

    models_and_their_labels = {"claude_sonnet_4": "Code",
                               "gpt_4o": "General",
                               "ministral_8b": "small"}

    prices_csv_path = f"{default_csv_dir}/prices_final.csv"
    stats_csv_path = f"{default_csv_dir}/stats_final.csv"
    configs = [(all_temp_models, 0.2), (models_with_ten_entries, 0.6), (all_temp_models, 1)]
    load_cached = True  # Set this flag to toggle load vs. recompute
    load_prices = True

    load_stats = True
    dfs = {}

    for list_, config_ in configs:
        experiment_suffix = f"temp{config_}"
        csv_path = f"{default_csv_dir}/cost_{experiment_suffix}.csv"

        if load_cached and os.path.exists(csv_path):
            #  Load previously saved DataFrame
            dfs[config_] = pd.read_csv(csv_path)
        else:
            # Recompute and save to CSV
            df = get_configuration_cost(root, list_, dataset_mapping, experiment_suffix, n_first=10)
            df.to_csv(csv_path, index=False)
            dfs[config_] = df

    print("Getting prices")
    if not load_prices:
        df_prices = compute_per_candidate_pricing(dataset_mapping, dfs, prices_csv_path)
    else:
        df_prices = pd.read_csv(prices_csv_path)

    print("Getting stats")
    if not load_stats:
        df_stats = compute_per_config_prices(df_prices, stats_csv_path)
    else:
        df_stats = pd.read_csv(stats_csv_path)

    print("Getting latex")
    merged_ablation_dict = merge_campaign_and_pricing_data(df_stats, default_csv_dir,
                                                           loaded_config_to_ablation_csv_mapping,
                                                           models_and_their_labels)

    generate_latex_table_ablation(merged_ablation_dict, candidate_thresholds, samples_thresholds, temp_thresholds,
                                  default_latex_dir, models_and_their_labels)
    print("Done")
