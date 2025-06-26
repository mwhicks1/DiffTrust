import os
import glob
import json
from shutil import copyfile

def filter_first_success_or_failure(data):
    seen = {}
    for entry in data:
        name = entry.get("name")
        is_success = "Dis" in entry and "Err" in entry

        if name not in seen:
            seen[name] = entry
        elif "Dis" in seen[name] and "Err" in seen[name]:
            continue  # Already have a success, skip
        elif is_success:
            seen[name] = entry  # Overwrite first failure with first success

    return list(seen.values())

if __name__ == "__main__":
    log_dirs = glob.glob(".experiments/*/logs")

    for log_dir in log_dirs:
        pointwise_file = os.path.join(log_dir, "experiment-pointwise.json")
        backup_file = os.path.join(log_dir, "experiment-pointwise-0100.json")

        if not os.path.exists(pointwise_file):
            print(f" Skipping {log_dir} — no experiment-pointwise.json found")
            continue

        # Backup original
        print(f"  Backing up {pointwise_file} → {backup_file}")
        copyfile(pointwise_file, backup_file)

        # Load, filter, and overwrite
        print(f"  Filtering {pointwise_file}")
        with open(pointwise_file) as f:
            raw_data = json.load(f)

        filtered_data = filter_first_success_or_failure(raw_data)

        with open(pointwise_file, "w") as f:
            json.dump(filtered_data, f, indent=2)

        print(f" Updated: {pointwise_file}\n")