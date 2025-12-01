#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from difftrust.rqs.utils import detection_metrics, make_result_dir
from difftrust.rqs.plots import save_log2_scatter_plot

# Load experiment results
experiment_file = sys.argv[1] if len(sys.argv) > 1 else '.experiments/AAAA/logs/experiment-pointwise.json'
with open(experiment_file, 'r') as f:
    results = json.load(f)

# Extract metrics (filter out failed records)
results = [r for r in results if 'Err' in r and 'Dis' in r]
errs = [r['Err'] for r in results]
diss = [r['Dis'] for r in results]
names = [r['name'] for r in results]

# Create output directory
csv_dir, plots_dir, tex_dir = make_result_dir('.experiments/AAAA/analysis')

# Calculate detection metrics
metrics = detection_metrics("AAAA", errs, diss)

# Print summary
print("\n=== Experiment Summary ===")
for key, value in metrics.items():
    print(f"{key}: {value}")

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(f"{csv_dir}/results.csv", index=False)

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(f"{csv_dir}/metrics.csv", index=False)

# Generate scatter plot
rho, _ = spearmanr(errs, diss)
save_log2_scatter_plot(
    plots_dir,
    x_vals=np.array(diss),
    y_vals=np.array(errs),
    alternate_label=f"ρ = {rho:.3f}",
    file_name="error_vs_incoherence.png"
)

print(f"\n✓ Results saved to .experiments/AAAA/analysis/")
print(f"  - CSV: {csv_dir}/")
print(f"  - Plots: {plots_dir}/")
