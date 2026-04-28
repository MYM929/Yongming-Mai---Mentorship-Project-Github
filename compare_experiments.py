import argparse
import csv
import math
import os

from experiment_config import collect_run_summary_paths, load_dataset_config


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

IMPORTANT_COLUMNS = [
    "timestamp",
    "active_dataset",
    "accepted_frame_ratio",
    "num_segments",
    "head_tail_translation_drift_m",
    "head_tail_rotation_drift_deg",
    "total_path_length_m",
    "icp_fallback_rate",
    "local_lc_accept_rate",
    "total_global_lc_added",
    "total_wall_time_sec",
    "runtime_per_accepted_frame_sec",
    "final_memory_mb",
    "overall_run_quality",
]

HIGHER_IS_BETTER = {
    "accepted_frame_ratio",
    "local_lc_accept_rate",
    "orb_glc_accept_rate",
    "fpfh_glc_accept_rate",
    "total_global_lc_added",
}

QUALITY_RANK = {
    "Excellent": 3,
    "Good": 2,
    "Needs Review": 1,
}


def load_default_experiments_dir():
    config = load_dataset_config()
    return config["dataset_experiments_dir"]


def to_float(value):
    try:
        if value is None or value == "":
            return None
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def sorted_rows(rows, sort_column=None):
    if not sort_column:
        return sorted(
            rows,
            key=lambda row: (
                -(to_float(row.get("accepted_frame_ratio")) or 0.0),
                to_float(row.get("head_tail_translation_drift_m"))
                if to_float(row.get("head_tail_translation_drift_m")) is not None
                else float("inf"),
                to_float(row.get("total_wall_time_sec"))
                if to_float(row.get("total_wall_time_sec")) is not None
                else float("inf"),
            ),
        )

    reverse = sort_column in HIGHER_IS_BETTER or sort_column == "overall_run_quality"
    has_numeric_values = any(to_float(row.get(sort_column)) is not None
                             for row in rows)

    if sort_column == "overall_run_quality":
        return sorted(
            rows,
            key=lambda row: QUALITY_RANK.get(row.get(sort_column, ""), 0),
            reverse=True,
        )

    if has_numeric_values:
        missing_value = float("-inf") if reverse else float("inf")
        return sorted(
            rows,
            key=lambda row: (
                to_float(row.get(sort_column))
                if to_float(row.get(sort_column)) is not None
                else missing_value
            ),
            reverse=reverse,
        )

    return sorted(rows, key=lambda row: str(row.get(sort_column, "")),
                  reverse=reverse)


def format_cell(value):
    numeric = to_float(value)
    if numeric is None:
        return value or ""
    if abs(numeric) >= 100:
        return f"{numeric:.1f}"
    if abs(numeric) >= 10:
        return f"{numeric:.2f}"
    return f"{numeric:.3f}"


def print_table(rows, columns):
    widths = {
        col: max(len(col), *(len(format_cell(row.get(col, ""))) for row in rows))
        for col in columns
    }
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    divider = "-+-".join("-" * widths[col] for col in columns)
    print(header)
    print(divider)
    for row in rows:
        print(" | ".join(format_cell(row.get(col, "")).ljust(widths[col])
                         for col in columns))


def save_markdown(rows, columns, path):
    lines = [
        "# Pose Trajectory Experiment Comparison",
        "",
        f"Total experiments: {len(rows)}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(col, ""))
                                       for col in columns) + " |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Compare pose trajectory experiment summary rows.")
    parser.add_argument("--sort", help="CSV column to sort by.")
    parser.add_argument(
        "--config",
        help="Config file path. Defaults to the config selected by config.json.",
    )
    parser.add_argument(
        "--experiments",
        help="Dataset experiment folder. Defaults to the active dataset experiments folder.",
    )
    args = parser.parse_args()

    experiments_dir = args.experiments or (
        load_dataset_config(args.config)["dataset_experiments_dir"]
        if args.config
        else load_default_experiments_dir()
    )
    summary_paths = collect_run_summary_paths(experiments_dir)
    if not summary_paths:
        raise FileNotFoundError(
            f"No experiment summaries found under {experiments_dir}"
        )

    rows = []
    for summary_path in summary_paths:
        with open(summary_path, "r", newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        print("No experiments found.")
        return

    if args.sort and args.sort not in rows[0]:
        available = ", ".join(rows[0].keys())
        raise ValueError(f"Unknown sort column '{args.sort}'. Available: {available}")

    rows = sorted_rows(rows, args.sort)
    columns = [col for col in IMPORTANT_COLUMNS if col in rows[0]]
    print_table(rows, columns)

    comparison_path = os.path.join(experiments_dir, "experiment_comparison.md")
    save_markdown(rows, columns, comparison_path)
    print(f"\nSaved comparison report to {comparison_path}")


if __name__ == "__main__":
    main()
