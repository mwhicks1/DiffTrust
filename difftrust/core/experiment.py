import json
import pathlib
from typing import Any

import cloudpickle


class ExperimentCtxt:

    def __init__(self, working_dir: str):
        self.working_dir = pathlib.Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = self.working_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logs_dir = self.working_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)  # Fix: was wrongly self.cache_dir again

        self.config_file = self.working_dir / "config.json"
        self.config = {}
        if self.config_file.exists():
            with open(self.config_file.as_posix(), "r", encoding='utf-8') as config_file:
                self.config = json.load(config_file)
        else:
            with open(self.config_file.as_posix(), "w", encoding='utf-8') as config_file:
                config_file.write(json.dumps(self.config, indent=2))

    def _update_config(self):
        with open(self.config_file.as_posix(), "w", encoding='utf-8') as config_file:
            config_file.write(json.dumps(self.config, indent=2))

    def set_config(self, key, value):
        self.config[key] = value
        self._update_config()

    def get_config(self, key):
        return self.config[key]

    def log(self, log_name: str, log_object):
        log_file = self.logs_dir / f"{log_name}.json"
        existing_logs = self.get_logs(log_name)
        existing_logs.append(log_object)
        with open(log_file.as_posix(), "w", encoding='utf-8') as f:
            f.write(json.dumps(existing_logs, indent=2))

    def get_logs(self, log_name: str):
        log_file = self.logs_dir / f"{log_name}.json"
        if log_file.exists():
            with open(log_file.as_posix(), "r", encoding='utf-8') as f:
                logs = json.load(f)
                if not isinstance(logs, list):
                    return [logs]
                return logs
        else:
            return []

    def cache_object(self, file_name: str, obj: Any):
        cache_file = self.cache_dir / file_name
        with open(cache_file.as_posix(), "wb") as f:
            cloudpickle.dump(obj, f)
        return cache_file

    def get_cache(self, file_name: str):
        cache_file = self.cache_dir / file_name
        if cache_file.exists():
            with open(cache_file.as_posix(), "rb") as f:
                return cloudpickle.load(f)
        return None
