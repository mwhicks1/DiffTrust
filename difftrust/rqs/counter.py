from collections import defaultdict
from pathlib import Path

from difftrust.core.experiment import ExperimentCtxt  # adjust import if needed

THRESHOLDS = [50, 10]  # sorted descending for proper counting


def count_candidates_by_threshold_sorted(benchmark_name: str, root_dir: str, config_results: dict):
    root_path = Path(root_dir)
    experiments_dir = root_path / ".experiments"

    if not experiments_dir.exists():
        print(f"No experiments directory found at {experiments_dir}")
        return

    for project_dir in experiments_dir.iterdir():
        if not project_dir.is_dir():
            continue

        cache_dir = project_dir / "cache"
        if not cache_dir.exists():
            continue

        ctxt = ExperimentCtxt(project_dir)
        threshold_counts = defaultdict(int)

        for file in cache_dir.iterdir():
            if file.name.startswith("candidates_"):
                try:
                    candidates = ctxt.get_cache(file.name)
                    if not isinstance(candidates, list):
                        continue
                    n = len(candidates)
                    for threshold in THRESHOLDS:
                        if n >= threshold:
                            threshold_counts[threshold] += 1
                            # break
                except Exception:
                    continue

        if threshold_counts:
            config = project_dir.name
            if config not in config_results:
                config_results[config] = {}
            config_results[config][benchmark_name] = {
                t: threshold_counts.get(t, 0) for t in THRESHOLDS
            }


def summarize_results(config_results):
    for config, benchmarks in sorted(config_results.items()):
        benchmark_summaries = []
        for benchmark_name, counts in benchmarks.items():
            count_str = " ".join(f"{k}: {v}" for k, v in counts.items())
            benchmark_summaries.append(f"{benchmark_name.lower()} {{{count_str}}}")
        summary = " ".join(benchmark_summaries)
        print(f"{config} → {summary}")


# Collect results
config_results = {}

count_candidates_by_threshold_sorted("HumanEval", "./HumanEval", config_results)
count_candidates_by_threshold_sorted("MBPP", "./MBPP", config_results)

# Print merged summary
print("\nMerged Summary:\n" + "-" * 100)
summarize_results(config_results)
