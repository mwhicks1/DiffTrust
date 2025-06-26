# DiffTrust: Paper Analysis Scripts (rqs/)

This folder contains the post-experiment analysis pipeline used in the DiffTrust paper. These scripts are responsible for aggregating results, generating all paper figures and tables, and conducting ablation and cost analyses.

For running the actual experiments and obtaining the raw logs, see the root [README.md](../README.md).

---

## Contents

- **`ablation.py`**  
  Performs controlled experiments to study the effect of varying parameters like the number of samples (`m`) or test inputs (`n`) on detection rate and incoherence/error alignment.

- **`costs_calculation.py`**  
  Estimates the monetary cost of querying various LLMs at scale, given pricing data and sampling configurations used in the experiments.

- **`counter.py`**  
  Tracks frequencies of outcomes like successful completions, failed compilations, or timeout-induced drops during batch analysis.

- **`constants.py`**  
  Stores global constants and mappings used across scripts, such as benchmark and model names, figure labels, and ordering for plots.

- **`custom_classes.py`**  
  Defines helper data structures for encapsulating parsed results and evaluation summaries in a structured way.

- **`latex.py`**  
  Converts numerical results and statistical outputs into formatted LaTeX tables used in the final paper submission.

- **`plots.py`**  
  Generates all figures from the paper, including:
  - Scatter plots of rank correlations
  - Disagreement/error alignment plots
  - Cost-performance tradeoff visualizations
  - Ridgeline and ablation curves

- **`rq_utils.py`**  
  Shared helper functions across the RQ scripts for filtering, normalization, rank computation, and aggregation.

- **`utils.py`**  
  General-purpose utilities for loading logs, parsng experiment outputs, and supporting I/O across formats.

---

## Reproducing Paper Figures & Tables

To regenerate the tables and figures from the DiffTrust paper, run the corresponding scripts directly after experiments are complete. Each script assumes that results are organized and cached in a structure expected by `ExperimentCtxt`.

> For full details on experiment configuration and log output, see the root `README.md`.

---

## Notes

- All figures and LaTeX tables are automatically saved to disk under a paper-structured output directory (e.g., `results_submission_test/`).
- If you are re-running or adapting experiments, adjust paths and constants in `constants.py`.

