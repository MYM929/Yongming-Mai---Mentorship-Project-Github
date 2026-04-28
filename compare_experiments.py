import argparse
import csv
import json
import math
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "dataset_config.json")
DATASETS_ROOT = os.path.join(SCRIPT_DIR, "datasets")

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


def load_active_reports_dir():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    active_dataset = config.get("active_dataset")
    if not active_dataset:
        raise ValueError("'active_dataset' is missing in dataset_config.json")
    return os.path.join(DATASETS_ROOT, active_dataset, "reports")


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
        "--reports",
        help="Reports folder. Defaults to the active dataset reports folder.",
    )
    args = parser.parse_args()

    reports_dir = args.reports or load_active_reports_dir()
    summary_path = os.path.join(reports_dir, "experiment_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing experiment summary: {summary_path}")

    with open(summary_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No experiments found.")
        return

    if args.sort and args.sort not in rows[0]:
        available = ", ".join(rows[0].keys())
        raise ValueError(f"Unknown sort column '{args.sort}'. Available: {available}")

    rows = sorted_rows(rows, args.sort)
    columns = [col for col in IMPORTANT_COLUMNS if col in rows[0]]
    print_table(rows, columns)

    comparison_path = os.path.join(reports_dir, "experiment_comparison.md")
    save_markdown(rows, columns, comparison_path)
    print(f"\nSaved comparison report to {comparison_path}")


if __name__ == "__main__":
    main()
