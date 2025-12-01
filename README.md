# DiffTrust: Estimating Correctness Without Oracles in LLM-Based Code Generation

This repository contains the implementation used in the experiments for the DiffTrust paper, which introduces *incoherence* as a theoretically grounded proxy for correctness in LLM-based code generation—designed to operate without access to ground-truth implementations or oracles.

## Fork notes

This repository is a fork of the original, whose README continues below. Changes were made so that LLMs can be executed via AWS Bedrock. Note that you must use Python 3.12 to run the experiments. The script `analyze_experiment.py` will generate an incoherence vs. error plot after running an experiment.

## Overview

Large Language Models (LLMs) have demonstrated strong performance in code generation tasks, yet concerns about *confabulation* remain—models frequently produce syntactically valid but semantically incorrect programs. In DiffTrust, we propose a principled proxy for correctness called **incoherence**, which quantifies semantic disagreement between independently sampled model outputs.

This repository supports empirical evaluation of our new metrics across two popular benchmarks: **HumanEval** and **MBPP**.

> **Note**: This repository is not intended as a general-purpose benchmarking toolkit, but rather as the exact implementation behind our experimental results.

## Paper Contributions (Recap)

1. **Incoherence** is proposed as an unsupervised, theoretically grounded estimator for correctness.
2. A **probabilistic framework** links incoherence to model error via a provable lower bound.
3. Empirical validation across 16 LLMs showing that incoherence alone can detect ~2/3 of incorrect progrms *without any false positives*, matching oracle-based rankings with Spearman’s ρ ≥ 0.92.

## Project Structure

```
.
├── HumanEval
│   ├── instance.py
│   ├── remove_duplicates.py
│   ├── run.py
│   └── stats.py
├── MBPP
│   ├── instance.py
│   ├── remove_duplicates.py
│   ├── run.py
│   └── stats.py
├── README.md
├── difftrust
│   ├── config.json
│   ├── config.py
│   ├── core
│   │   ├── checking.py
│   │   ├── coder.py
│   │   ├── experiment.py
│   │   ├── function.py
│   │   ├── metrics.py
│   │   ├── refiner.py
│   │   └── specification.py
│   ├── fuzzer
│   │   ├── coverage.py
│   │   └── fuzzer.py
│   ├── generic
│   │   ├── generic_equal.py
│   │   ├── generic_explorer.py
│   │   ├── generic_fuzzer.py
│   │   ├── generic_mutator.py
│   │   └── generic_repr.py
│   ├── llm
│   │   ├── abstract.py
│   │   ├── chat.py
│   │   ├── chatgpt.py
│   │   ├── claude.py
│   │   ├── gemini.py
│   │   ├── open_router.py
│   │   └── open_router_models.json
│   ├── rqs
│   │   ├── ablation.py
│   │   ├── aggregate-plots
│   │   ├── constants.py
│   │   ├── costs_calculation.py
│   │   ├── counter.py
│   │   ├── custom_classes.py
│   │   ├── latex.py
│   │   ├── plots.py
│   │   ├── rq_utils.py
│   │   └── utils.py
│   └── tracing
│       ├── events.py
│       └── tracer.py
└── requirements.txt

```

## Requirements

Install dependencies via:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

Each dataset folder (e.g., `MBPP/` or `HumanEval/`) contains a `run.py` script that replicates the experimental setup from the paper.

### Example

```bash
# Run (from the root directory) the pointwise incoherence experiment on HumanEval
python -m HumanEval.run
```

To adjust parameters such as the LLM, number of candidate functions (`nb_candidate`), number of test inputs (`nb_sample`), or temperature settings.

### Key Parameters (set in `run.py`)
- `llm_name`: Name of the LLM to use (e.g., "gpt_4", "claude_opus_4", etc.)
- `nb_candidate`: Number of candidate programs to sample per task (default: 10)
- `nb_sample`: Number of test inputs per comparison (default: 1000)
- `temperature`: Sampling temperature for the LLM (default: 0.0 for deterministic outputs)
- `timeout`: Max execution time per comparison (default: 60 seconds)


[//]: # (## Citation)

[//]: # ()
[//]: # (> 📄 *Citation will be added once the paper is published.*)

## License

MIT License. See [LICENSE](./LICENSE) for ful text.
