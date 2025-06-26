def append_temp_to_name(model_names, temp):
    result = [f"{model_name}_temp{temp}" for model_name in model_names]
    return result


reasoning_models = {"gpt_o4_mini", 'gemini_2_5_pro', 'gemini_2_5_flash',
                    'deepseek_r1', 'claude_opus_4', 'claude_sonnet_4',
                    "llama_4_maverick_17b",
                    "gpt_4", }
general_purpose_models = {'gpt_4o', 'deepseek_v3_0324', "llama_3_3_70b_instruct",
                          'gpt_4_turbo', "gpt_3_5"}
small_mid_size_modelz = {"gemini_2_0_flash_lite", "ministral_8b", "llama_3_1_8b_instruct"}

models_with_ten_entries = {'claude_opus_4', 'claude_sonnet_4', "deepseek_r1", "deepseek_v3_0324",
                           "gemini_2_0_flash_lite", "gemini_2_5_flash",
                           "gemini_2_5_pro", "gpt_3_5", "gpt_4", "gpt_4_turbo",
                           "gpt_4o", "gpt_o4_mini", "llama_3_1_8b_instruct",
                           "llama_3_3_70b_instruct", "llama_4_maverick_17b", "ministral_8b"}
models_with_fifty_entries = {"deepseek_v3_0324", 'claude_sonnet_4', "gpt_4o",
                             "llama_4_maverick_17b", "ministral_8b"}
models_with_fifty_for_06_only = models_with_fifty_entries.union({"gemini_2_5_flash"})
models_for_comparison_table = ['gemini_2_5_pro', 'gpt_4o', "ministral_8b"]
all_temp_models = {'claude_opus_4', 'claude_sonnet_4', "deepseek_v3_0324",
                   "gemini_2_5_flash", "gpt_3_5", "gpt_4o",
                   "llama_4_maverick_17b", "ministral_8b"}

ablation_models = {"claude_opus_4", "claude_sonnet_4", "gemini_2_5_flash", "gpt_4o", "llama_4_maverick_17b",
                   }

MODEL_PRETTY_NAME = {
    # OpenAI
    "gpt_4": "GPT-4",
    "gpt_4_turbo": "GPT-4 Turbo",
    "gpt_3_5": "GPT-3.5",
    "gpt_4o": "GPT-4o",
    "gpt_o4_mini": "GPT-o4 Mini",

    # Google Gemini
    "gemini_2_5_pro": "Gemini 2.5 Pro",
    "gemini_2_5_flash": "Gemini 2.5 Flash",
    "gemini_2_0_flash_lite": "Gemini 2.0 Flash Lite",

    # Anthropic Claude
    "claude_opus_4": "Claude 4 Opus",
    "claude_sonnet_4": "Claude 4 Sonnet",

    # DeepSeek
    "deepseek_r1": "DeepSeek-Coder R1",
    "deepseek_v3_0324": "DeepSeek-V3 (Mar 2024)",

    # Meta LLaMA
    "llama_3_3_70b_instruct": "LLaMA 3 70B Instruct",
    "llama_3_1_8b_instruct": "LLaMA 3 8B Instruct",
    "llama_4_maverick_17b": "LLaMA 4 Maverick 17B",

    # Mistral
    "ministral_8b": "Mistral 8B",
}

ablation_models = all_temp_models

# Load data
model_groups_lists = [reasoning_models, general_purpose_models, small_mid_size_modelz]
models_list = [model_ for model_list in model_groups_lists for model_ in model_list]
# call append_temp_to_name() repeatedly and update the results to have all the combos
list_with_temp_appended_06_10 = append_temp_to_name(models_with_ten_entries, .6)
# get list of 0.2 temps for 10
list_with_temp_appended_02_10 = append_temp_to_name([i for i in all_temp_models], 0.2)

#  get list of 1 temps for 10
list_with_temp_appended_1_10 = append_temp_to_name([i for i in all_temp_models], 1)

# call append_temp_to_name() repeatedly and update the results to have all the combos
list_with_temp_appended_06_50 = append_temp_to_name([i for i in models_with_fifty_entries if i in all_temp_models], .6)
list_with_temp_appended_06_50.extend(
    append_temp_to_name([i for i in models_with_fifty_for_06_only if i in all_temp_models], .6))
# get list of 0.2 temps for 50
list_with_temp_appended_02_50 = append_temp_to_name([i for i in models_with_fifty_entries if i in all_temp_models], .2)

# get list of 1 temps for 50
list_with_temp_appended_1_50 = append_temp_to_name([i for i in models_with_fifty_entries if i in all_temp_models], 1)

overrides = {}
ablation_temps = [0.2, 0.6, 1]
main_ranges = [10, 50]
ablation_candidate_range = list(range(1, 51))
ablation_samples_range = [100]
ablation_samples_range.extend(list(range(1000, 10500, 500)))
# ablation_samples_range.append(100000)
ablation_comparisons = [("Raw Dis Mean Rank", "Raw Err Mean Rank"),
                        ("Error Rate (Errors / Total) Rank", "Detection Rate (Detected / Errors) Rank")]

# Pricing structure: (prompt_price_per_1k, completion_price_per_1k)
model_pricing = {
    "gpt_4": (30.0, 60.0),
    "gpt_4_turbo": (10.0, 30.0),
    "gpt_3_5": (1.0, 2.0),
    "gpt_4o": (5.0, 15.0),
    "gpt_o4_mini": (1.1, 4.4),
    "claude_opus_4": (15.0, 75.0),
    "claude_sonnet_4": (3.0, 15),
    "gemini_2_5_pro": (1.25, 10.0),
    "gemini_2_5_flash": (0.3, 2.5),
    "gemini_2_0_flash_lite": (0.075, 0.3),
    "deepseek_r1": (0.45, 2.15),
    "deepseek_v3_0324": (0.3, 0.88),
    "llama_4_maverick_17b": (0.15, 0.6),
    "llama_3_3_70b_instruct": (0.05, 0.25),
    "llama_3_1_8b_instruct": (0.016, 0.03),
    "ministral_8b": (0.01, 0.01),
}

config_to_ablation_csv_mapping = \
    {"candidate":
        {
            "MBPP": "ablation_df_rq3_1_06_50_candidate_MBPP.csv",
            "HumanEval": "ablation_df_rq3_1_06_50_candidate_HumanEval.csv"
        },
        "sample":
            {
                "MBPP": "ablation_df_rq3_2_sample_MBPP.csv",
                "HumanEval": "ablation_df_rq3_2_sample_HumanEval.csv"
            },
        "temp":
            {
                "MBPP": "ablation_df_rq3_3_temp_MBPP.csv",
                "HumanEval": "ablation_df_rq3_3_temp_HumanEval.csv"
            }

    }

candidate_thresholds = [1, 2, 5, 10, 25, 50]
samples_thresholds = [100, 1000, 2000, 5000, 10000]
temp_thresholds = [0.2, 0.6, 1]


def normalize_model_name(model: str) -> str:
    return model.replace("gpt_", "gpt-").replace("gpt_o4_mini", "gpt-3.5-turbo").replace("gpt-o4_mini",
                                                                                         "gpt-3.5-turbo").replace("3_5",
                                                                                                                  "3.5").replace(
        'gpt_4_turbo', 'gpt-4').replace('gpt-4_turbo', 'gpt-4')


# Approximate character-per-token divisors for prompt and completion
model_token_divisors = {
    # OpenAI models (≈4 chars/token)
    "gpt_4": (4.0, 4.0),
    "gpt_4_turbo": (4.0, 4.0),
    "gpt_3_5": (4.0, 4.0),
    "gpt_4o": (4.0, 4.0),
    "gpt_o4_mini": (4.0, 4.0),

    # Anthropic models (≈3.5 chars/token)
    "claude_opus_4": (3.5, 3.5),
    "claude_sonnet_4": (3.5, 3.5),

    # Gemini models (≈4.0 chars/token)
    "gemini_2_5_pro": (4.0, 4.0),
    "gemini_2_5_flash": (4.0, 4.0),
    "gemini_2_0_flash_lite": (4.0, 4.0),

    # DeepSeek + LLaMA models (≈3.75 chars/token)
    "deepseek_r1": (3.75, 3.75),
    "deepseek_v3_0324": (3.75, 3.75),
    "llama_4_maverick_17b": (3.75, 3.75),
    "llama_3_3_70b_instruct": (3.75, 3.75),
    "llama_3_1_8b_instruct": (3.75, 3.75),

    # Mistral (≈3.75 chars/token)
    "ministral_8b": (3.75, 3.75),
}
