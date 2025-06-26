from pylatex import Document, Section, Subsection, Figure, Tabular, NewLine, NoEscape, NewPage
import os
import matplotlib.pyplot as plt
import numpy as np
from pylatex.utils import bold
from scipy.stats import pearsonr, spearmanr
from difftrust import experiment
import pathlib


def make_report(ctxt_exp_name: str):

    report_dir = pathlib.Path(ctxt_exp_name).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = (report_dir / "plot")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ctxt = experiment.ExperimentCtxt(ctxt_exp_name)
    logs = ctxt.get_logs("experiment-pointwise")

    # Fetch experiment metadata
    llm_name = ctxt.get_config("llm")
    number_candidate = ctxt.get_config("nb_candidate")
    number_tests = ctxt.get_config("nb_sample")
    temperature = ctxt.get_config("temperature")
    timeout = ctxt.get_config("timeout")

    # Fetch data
    raw_incoh = [log.get("Dis", 0.0) for log in logs]
    raw_err = [log.get("Err", 0.0) for log in logs]
    total = len(logs)

    # Generate bubble plot
    save_bubble_plot(raw_incoh, raw_err, "Raw", "viridis", plot_dir)
    save_loglog_plot(raw_incoh, raw_err, plot_dir)

    # --- LaTeX Report ---
    doc = Document((report_dir / "report").as_posix(), document_options=["12pt"])

    with doc.create(Section("Incoherence-Based Experiment Analysis")):
        doc.append(
            "This report presents a statistical analysis of the model’s performance across tasks, "
            "focusing on the relationship between incoherence scores (Incoherence) and execution errors (Error)."
        )
        doc.append(NewLine())
        doc.append(NewLine())
        doc.append(f"Number of tasks analyzed: {total}")

        with doc.create(Section("Introduction")):
            doc.append(
                "This report summarizes the results of an automatic evaluation of code generation using the following "
                "configuration parameters.")
            doc.append(NewLine())

            with doc.create(Tabular("ll")) as table:
                table.add_row([bold("Parameter"), bold("Value")])
                table.add_hline()
                table.add_row(["Language Model", llm_name])
                table.add_row(["Temperature", str(temperature)])
                table.add_row(["$m$ (number of candidates)", str(number_candidate)])
                table.add_row(["$n$ (number of samples used to estimate metrics)", str(number_tests)])
                table.add_row(["Timeout per metric estimation (s)", str(timeout)])

            doc.append(NewLine())
            doc.append(
                "The model was tested across a suite of programming tasks. "
                "We aim to explore how the model’s incoherence signal relates to execution-time failures."
            )
        with doc.create(Subsection("Summary Statistics")):
            stats = [
                ("Raw Incoherence", raw_incoh),
                ("Raw Error", raw_err)
            ]
            with doc.create(Tabular("lcccc")) as table:
                table.add_row(["Metric", "Mean", "Std", "Min", "Max"])
                for label, values in stats:
                    if not values:
                        table.add_row([label, "N/A", "N/A", "N/A", "N/A"])
                    else:
                        table.add_row([
                            label,
                            f"{np.mean(values):.3f}",
                            f"{np.std(values):.3f}",
                            f"{np.min(values):.3f}",
                            f"{np.max(values):.3f}"
                        ])

        with doc.create(Subsection("Error Detection Analysis")):
            def detection_metrics(errs, incohs):
                total = len(errs)
                errors = sum(1 for i in range(total) if errs[i] > 0.0)
                detected = sum(1 for i in range(total) if errs[i] > 0.0 and incohs[i] > 0.0)
                confident = sum(1 for i in range(total) if incohs[i] == 0.0)
                confident_errors = sum(1 for i in range(total) if errs[i] > 0.0 and incohs[i] == 0.0)
                mean_when_confident = (
                    sum(errs[i] for i in range(total) if incohs[i] == 0.0) / confident
                    if confident_errors > 0 else 0.0
                )
                return {
                    "Errors (Error > 0)": errors,
                    "Error Rate": f"{errors / total:.2%}",
                    "Detected Errors (Error > 0 and Incoherence > 0)": detected,
                    "Detection Rate": f"{detected / errors:.2%}" if errors else "-",
                    "Confident (Incoherence = 0)": confident,
                    "Confident Error Count": confident_errors,
                    "Confident Error Rate": f"{confident_errors / confident:.2%}" if confident else "-",
                    "Mean Error When Confident": f"{mean_when_confident:.4f}"
                }

            metrics = detection_metrics(raw_err, raw_incoh)
            with doc.create(Tabular("lc")) as table:
                table.add_row(["Metric", "Value"])
                for key, val in metrics.items():
                    table.add_row([key, str(val)])

        with doc.create(Subsection("Correlation Analysis")):
            def correlation(x, y):
                p_corr, p_pval = pearsonr(x, y)
                s_corr, s_pval = spearmanr(x, y)
                return f"{p_corr:.3f}", f"{p_pval:.3e}", f"{s_corr:.3f}", f"{s_pval:.3e}"

            with doc.create(Tabular("lcccc")) as table:
                table.add_row(["Metric", "Pearson r", "Pearson p", "Spearman $\\rho$", "Spearman p"])
                table.add_row(["Incoherence vs Error"] + list(correlation(raw_incoh, raw_err)))

        doc.append(NewPage())

        with doc.create(Subsection("Bubble Plot of Incoherence and Error")):
            doc.append(
                "This plot shows the density of (Incoherence, Error) points using bubble size to indicate frequency.")
            with doc.create(Figure(position='htbp')) as fig:
                fig.add_image((plot_dir / "bubble.png").as_posix(), width=NoEscape(r'0.75\linewidth'))
                fig.add_caption("Bubble Plot: Incoherence vs Error")

        with doc.create(Subsection("Log-Log Plot of Incoherence and Error")):
            doc.append("This plot displays the relationship between Incoherence and Error in log-log scale. "
                       "Only data points where both values are strictly positive are included.")
            with doc.create(Figure(position='htbp')) as fig:
                fig.add_image((plot_dir / "log_log.png").as_posix(), width=NoEscape(r'0.75\linewidth'))
                fig.add_caption("Log-Log Scatter Plot: Incoherence vs Error")

    # Compile LaTeX to PDF
    doc.generate_pdf(clean_tex=False, compiler='pdflatex')
    print("✅ PDF report generated as 'experiment_report_incoherence.pdf'")


# --- Bubble plot only ---
def save_bubble_plot(x_vals, y_vals, label, cmap, plot_dir):
    precision = 25
    bubble_data = {}
    for x, y in zip(x_vals, y_vals):
        x = round(precision * x) / precision
        y = round(precision * y) / precision
        bubble_data[(x, y)] = bubble_data.get((x, y), 0) + 1

    bubble_plot = {'x': [], 'y': [], 'size': [], 'color': []}
    for (x, y), count in bubble_data.items():
        bubble_plot['x'].append(x)
        bubble_plot['y'].append(y)
        bubble_plot['size'].append(50 * count)
        bubble_plot['color'].append(count)

    plt.figure(figsize=(8, 6))
    plt.scatter(bubble_plot['x'], bubble_plot['y'],
                s=bubble_plot['size'],
                c=bubble_plot['color'],
                cmap=cmap, alpha=0.6, edgecolor='k')

    # Add dashed reference line: incoherence = 2 * error
    x_range = np.linspace(0, max(bubble_plot['x']) * 1.05, 500)
    y_line = 0.5 * x_range
    plt.plot(x_range, y_line, linestyle='--', color='gray', label='Incoherence = 2 × Error')

    plt.colorbar(label="Frequency")
    plt.xlabel(f"{label} Incoherence (rounded)")
    plt.ylabel(f"{label} Error (rounded)")
    plt.title(f"{label} Error vs Incoherence - Bubble Plot")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig((plot_dir / "bubble.png").as_posix())
    plt.close()


# --- Log-Log plot ---
def save_loglog_plot(x_vals, y_vals, plot_dir):
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    mask = (x_vals > 0) & (y_vals > 0)
    x_log = x_vals[mask]
    y_log = y_vals[mask]

    if len(x_log) == 0:
        print("No valid data for log-log plot (all values zero or negative).")
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(x_log, y_log, alpha=0.6, edgecolor='k')

    # Reference line: y = 0.5 * x (same slope in log-log)
    x_line = np.logspace(np.log10(min(x_log) * 0.8), np.log10(max(x_log) * 1.2), 500)
    y_line = 0.5 * x_line
    plt.plot(x_line, y_line, linestyle='--', color='gray', label='Incoherence = 2 × Error')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Incoherence (log scale)")
    plt.ylabel("Error (log scale)")
    plt.title("Log-Log Plot: Incoherence vs Error")
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig((plot_dir / "log_log.png").as_posix())
    plt.close()


if __name__ == "__main__":
    folder_pathes = [
        ".experiments",
        ".experiments-ablation",
        ".experiments-ablation-samples",
        ".experiments-ablation-temps"
    ]
    to_suppress = []
    for folder_path in folder_pathes:
        folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        for folder in folders:
            print(f"{folder_path}/{folder}")
            try:
                make_report(f"{folder_path}/{folder}")
            except Exception as exception:
                to_suppress.append(f"{folder_path}/{folder}")
    print(f"Experiments to suppress : {to_suppress}")
