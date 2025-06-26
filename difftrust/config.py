from json import load as _json_load
from pathlib import Path as _pathlib_path


def config():
    config_path = _pathlib_path(__file__).parent / "config.json"
    with open(config_path.as_posix(), 'r', encoding='utf-8') as f:
        return _json_load(f)
