"""Summarize lateral connection ablation runs (OFF vs ON) across seeds.

Reads results directories and outputs a clean comparison table with:
- Task-IL Accuracy (mean ± std)
- Task-IL Backward Transfer (BWT)
- Task-IL Forward Transfer (FWT)
- Class-IL NLL
- Final Lateral Magnitude (||Lambda||_F)
"""

import argparse
import glob
import json
import os
import numpy as np


def parse_runs(results_dir="results"):
    modes = ["off", "on"]
    data = {"off": [], "on": []}

    for mode in modes:
        pattern = os.path.join(results_dir, f"causal_coding_lat_{mode}_seed_*", "experiment_summary.json")
        for filepath in sorted(glob.glob(pattern)):
            if "_smoke" in filepath:
                continue
            with open(filepath, "r") as f:
                item = json.load(f)
                data[mode].append(item)

    return data


def format_stat(values, is_percent=True):
    if not values:
        return "N/A"
    arr = np.array(values)
    scale = 100.0 if is_percent else 1.0
    mean = np.mean(arr) * scale
    std = np.std(arr) * scale
    if is_percent:
        return f"{mean:.2f} ± {std:.2f}%"
    return f"{mean:.4f} ± {std:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Summarize Lateral Ablation Results")
    parser.add_argument("--results-dir", default="results", help="Path to results directory")
    args = parser.parse_args()

    data = parse_runs(args.results_dir)

    print("\n" + "=" * 80)
    print("LATERAL CONNECTION ABLATION EXPERIMENT SUMMARY")
    print("=" * 80)

    n_off = len(data["off"])
    n_on = len(data["on"])
    print(f"Runs found: Lateral OFF: {n_off} runs | Lateral ON: {n_on} runs\n")

    if n_off == 0 and n_on == 0:
        print(f"No completed full runs found in '{args.results_dir}'.")
        print("Run full experiments first via causal_coding/experiments/run_lateral_ablation.py")
        return

    metrics_def = [
        ("Task-IL Accuracy", lambda x: x["task_il_accuracy"], True),
        ("Task-IL BWT", lambda x: x["task_il_bwt"], True),
        ("Task-IL FWT", lambda x: x["task_il_fwt"], True),
        ("Class-IL NLL", lambda x: x["class_il_nll"], False),
        ("Final ||Lambda||_F", lambda x: x.get("final_lambda_frob", 0.0), False),
    ]

    print(f"{'Metric':<24} | {'Lateral OFF (A)':<20} | {'Lateral ON (B)':<20} | {'Difference (B - A)':<20}")
    print("-" * 92)

    for name, extractor, is_pct in metrics_def:
        vals_off = [extractor(r) for r in data["off"]]
        vals_on = [extractor(r) for r in data["on"]]

        str_off = format_stat(vals_off, is_pct)
        str_on = format_stat(vals_on, is_pct)

        if vals_off and vals_on:
            scale = 100.0 if is_pct else 1.0
            diff = (np.mean(vals_on) - np.mean(vals_off)) * scale
            diff_str = f"{diff:+.2f}%" if is_pct else f"{diff:+.4f}"
        else:
            diff_str = "N/A"

        print(f"{name:<24} | {str_off:<20} | {str_on:<20} | {diff_str:<20}")

    print("=" * 92 + "\n")


if __name__ == "__main__":
    main()
