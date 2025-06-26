from difftrust.rqs.constants import MODEL_PRETTY_NAME


def generate_small_table_of_comparison_latex(results, threshold, file_name, models):
    df_mbpp, df_humaneval = results["MBPP"][threshold][0], results["HumanEval"][threshold][0]
    latex_string = generate_latex_rq1_table(df_mbpp, df_humaneval, models)
    save_latex_to_file(latex_string, file_name)


def generate_latex_table_ablation(ablation_dict, candidate_thresholds, sample_thresholds, temp_thresholds, latex_root,
                                  model_name_mapping):
    todo_note = "\\todo{Add Capption}"
    candidate_ablation_table = generate_latex_candidate_ablation_table(ablation_dict['candidate'], candidate_thresholds,
                                                                       'no_candidates', "Num", "Cand.", todo_note,
                                                                       'rq3_1_candidate_ablation', model_name_mapping,
                                                                       "Candidate Function Ablation",
                                                                       "Temperature = 0.6, Test Inputs = 1000")
    save_latex_to_file(candidate_ablation_table, f"{latex_root}/rq3_1_candidate_ablation.tex")
    sample_ablation_table = generate_latex_candidate_ablation_table(ablation_dict['sample'], sample_thresholds,
                                                                    'threshold', "Num", "Inputs.", todo_note,
                                                                    'rq3_2_sample_ablation', model_name_mapping,
                                                                    "Input Size Ablation",
                                                                    "Temperature = 0.6, Candidate Functions = 10")
    save_latex_to_file(sample_ablation_table, f"{latex_root}/rq3_2_sample_ablation.tex")
    temp_ablation_table = generate_latex_candidate_ablation_table(ablation_dict['temp'], temp_thresholds,
                                                                  'threshold', "", "Temp", todo_note,
                                                                  'rq3_3_temp_ablation', model_name_mapping,
                                                                  "Temperature Ablation",
                                                                  "Test Inputs = 1000, Candidate Functions = 10")
    save_latex_to_file(temp_ablation_table, f"{latex_root}/rq3_3_temp_ablation.tex")


def generate_full_table_of_comparison_latex(results, threshold, file_name, models):
    columns = [
        "Raw Err Mean",
        "Raw Dis Mean",
        "Spearman Rho err vs. diss",
        "Detection Rate (Detected / Errors)",
        "Error Mean When Confident"
    ]
    df_mbpp, df_humaneval = results["MBPP"][threshold][0], results["HumanEval"][threshold][0]
    latex_string = generate_latex_rq1_table(df_mbpp, df_humaneval, models, show_mean=True, num_latex_columns=2,
                                            columns=columns, table_ref="tab:rq1_full")
    save_latex_to_file(latex_string, file_name)


def save_latex_to_file(latex_string, file_name):
    if not file_name.endswith(".tex"):
        file_name += ".tex"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(latex_string)


def generate_latex_rq1_table(df_mbpp, df_humaneval, models_order, show_mean=False, columns=None, num_latex_columns=1,
                             table_ref="tab:rq1"):
    if columns is None:
        columns = [
            "Raw Err Mean",
            "Raw Dis Mean",
            "Spearman Rho err vs. diss",
            "Detection Rate (Detected / Errors)",
        ]

    def summarize(df):
        df = df[df["model"].isin(models_order)].copy()
        df = df.set_index("model").loc[models_order]
        return df[columns]

    mbpp_summary = summarize(df_mbpp)
    humaneval_summary = summarize(df_humaneval)

    def row(model, row_):
        pretty_name = MODEL_PRETTY_NAME.get(model, model.replace("_", " ").capitalize())
        return f"{pretty_name} & " + " & ".join(f"{row_[col]:.4f}" for col in columns) + r"\\"

    def mean_row(label, summary_df):
        return f"\\textbf{{{label} (Mean)}} & " + " & ".join(
            f"{summary_df[col].mean():.4f}" for col in columns
        ) + r"\\"

    latex_env = f"\\begin{{table*}}" if num_latex_columns == 2 else f"\\begin{{table}}"
    end_env = f"\\end{{table*}}" if num_latex_columns == 2 else f"\\end{{table}}"

    header = " & ".join([r"\textbf{Model}"] + [f"\\textbf{{{col}}}" for col in columns]) + r"\\"
    lines = [
        f"{latex_env}\\scriptsize",
        r"\begin{tabular}{" + "l" * (len(columns) + 1) + "}",
        r"\toprule",
        header,
        r"\midrule"
    ]

    lines.append(r"\textbf{MBPP} & " + " & " * len(columns) + r"\\")
    for model in models_order:
        lines.append(row(model, mbpp_summary.loc[model]))

    if show_mean:
        lines.append(mean_row("MBPP", mbpp_summary))

    lines.append(r"\midrule")
    lines.append(r"\textbf{HumanEval} & " + " & " * len(columns) + r"\\")

    for model in models_order:
        lines.append(row(model, humaneval_summary.loc[model]))

    if show_mean:
        lines.append(mean_row("HumanEval", humaneval_summary))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\todo{Add caption.}}",
        f"\\label{{{table_ref}}}",
        end_env
    ]
    return "\n".join(lines)


def generate_latex_candidate_ablation_table(df, thresholds_, threshold_df_col_header, ablation_top_header,
                                            ablation_bottom_header, caption, label, model_name_mapping, title,
                                            tag_line):
    datasets = ["MBPP", "HumanEval"]
    # Sort dataframe for consistent output
    df = df.sort_values(by=["dataset", "model", "no_candidates"])

    # datasets = df["dataset"].unique()
    models = df["model"].unique()

    output = []

    output.append(r"\begin{table}\scriptsize\centering")
    output.append(r"\begin{tabular}{@{}c@{ }c@{ }l@{\quad}c@{ \ }@{ \ }|c@{\quad}c@{\quad}c@{\quad}c@{ }|@{}}")
    output.append(r"\toprule")
    output.append(r"\multicolumn{8}{c}{")
    output.append(r"  \makecell{")
    output.append(rf"    \textbf{{{title}}} \\")
    output.append(rf"    {tag_line}")
    output.append(r"  }")
    output.append(r"} \\")
    output.append(r"\cline{3-8}")
    output.append(rf"&&&\textbf{{{ablation_top_header}}}")
    output.append(r" &\textbf{Camp.} &\textbf{Query} & \textbf{Detection}& \textbf{Und. Mean}\\")
    output.append(r"&&\textbf{LLM} &")
    output.append(rf"\textbf{{{ablation_bottom_header}}}")
    output.append(r"& \textbf{Cost} & \textbf{Cost} 	& \textbf{Rate} & \textbf{Error}\\\hline")

    cols_to_keep = ['config_total_cost', 'config_mean_per_query_cost', 'Detection Rate (Detected / Errors)',
                    'Error Mean When Confident']

    for i, dataset in enumerate(datasets):
        dataset_parts = dataset.split(" ", 1)
        top = dataset_parts[0]
        bottom = dataset_parts[1] if len(dataset_parts) > 1 else ""
        if i > 0:
            output.append(r"\hline\hline")
        output.append(
            rf"\multirow{{{len(thresholds_) * 3}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\textbf{{{top}}}}}}}&\multirow{{18}}{{*}}{{\rotatebox[origin=c]{{90}}{{\textbf{{{bottom}}}}}}}")
        for mi, model in enumerate(models):
            start_and = "&" if mi == 0 else "&&"
            model_ = model if model not in model_name_mapping else model_name_mapping[model]
            output.append(rf"{start_and}  \multirow{{{len(thresholds_)}}}{{*}}{{{model_}}} & {thresholds_[0]} & " +
                          " & ".join(
                              f"{df[(df['dataset'] == dataset) & (df['model'] == model) & (df[threshold_df_col_header] == thresholds_[0])][col].values[0]:.4f}"
                              for col in cols_to_keep) + r"\\")
            for size in thresholds_[1:]:
                output.append(r"&&& " + str(size) + " & " +
                              " & ".join(
                                  f"{df[(df['dataset'] == dataset) & (df['model'] == model) & (df[threshold_df_col_header] == size)][col].values[0]:.4f}"
                                  for col in cols_to_keep) + r"\\")

            output.append(r"\cmidrule{4-8}")
    output.append(r"\hline")
    output.append(r"\end{tabular}")
    output.append(rf"\caption{{{caption}}}")
    output.append(rf"\label{{{label}}}")
    output.append(r"\end{table}")
    return "\n".join(output)
