import json
import os
import re
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
DEFAULT_CONFIGS_DIR = os.path.join(SCRIPT_DIR, "configs")
DATASETS_ROOT = os.path.join(SCRIPT_DIR, "datasets")
EXPERIMENTS_ROOT = os.path.join(SCRIPT_DIR, "experiments")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_relative_to_script(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(SCRIPT_DIR, path_value))


def resolve_config_path(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    for index, arg in enumerate(argv):
        if arg == "--config":
            if index + 1 >= len(argv):
                raise ValueError("Expected a path after --config")
            return _resolve_relative_to_script(argv[index + 1])
        if arg.startswith("--config="):
            return _resolve_relative_to_script(arg.split("=", 1)[1])

    if os.path.exists(ROOT_CONFIG_PATH):
        root_config = _load_json(ROOT_CONFIG_PATH)
        config_path = root_config.get("config_path")
        if config_path:
            return _resolve_relative_to_script(config_path)

    raise FileNotFoundError(
        "Missing config selection. Set 'config_path' in config.json or pass --config."
    )


def load_dataset_config(config_path=None):
    config_path = os.path.normpath(config_path or resolve_config_path())
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file: {config_path}")

    config = _load_json(config_path)
    dataset_name = config.get("dataset_name") or config.get("active_dataset")
    dataset_dir_value = config.get("dataset_dir") or config.get("dataset_path")

    if dataset_dir_value:
        dataset_dir = _resolve_relative_to_script(dataset_dir_value)
        if not dataset_name:
            dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    else:
        if not dataset_name:
            raise ValueError(
                f"'dataset_name' is missing or empty in {config_path}"
            )
        dataset_dir = os.path.join(DATASETS_ROOT, dataset_name)

    if not dataset_name:
        raise ValueError(f"Could not determine dataset name from {config_path}")
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Configured dataset folder does not exist: {dataset_dir}"
        )

    return {
        "config_path": config_path,
        "dataset_name": dataset_name,
        "dataset_dir": dataset_dir,
        "dataset_experiments_dir": os.path.join(EXPERIMENTS_ROOT, dataset_name),
        "raw_config": config,
    }


def create_next_run_dir(dataset_experiments_dir):
    os.makedirs(dataset_experiments_dir, exist_ok=True)
    run_numbers = []
    for entry in os.listdir(dataset_experiments_dir):
        match = re.fullmatch(r"run(\d+)", entry)
        if match and os.path.isdir(os.path.join(dataset_experiments_dir, entry)):
            run_numbers.append(int(match.group(1)))
    next_run = max(run_numbers, default=0) + 1
    run_dir = os.path.join(dataset_experiments_dir, f"run{next_run}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def collect_run_summary_paths(dataset_experiments_dir):
    if not os.path.isdir(dataset_experiments_dir):
        return []
    summary_paths = []
    for entry in sorted(os.listdir(dataset_experiments_dir)):
        if not re.fullmatch(r"run\d+", entry):
            continue
        summary_path = os.path.join(
            dataset_experiments_dir, entry, "experiment_summary.csv"
        )
        if os.path.exists(summary_path):
            summary_paths.append(summary_path)
    return summary_paths
