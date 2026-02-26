"""Analyze W&B sweep results for assignment section 2.2.

Usage:
    python scripts/analyze_sweep_2_2.py --entity <wandb_entity> --project <project> --sweep-id <id>
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze W&B sweep for section 2.2")
    parser.add_argument("--entity", required=True, help="W&B entity (username/team)")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--sweep-id", required=True, help="W&B sweep id")
    # Changed default to 3 to align with Section 2.10 of the assignment
    parser.add_argument("--top-k", type=int, default=3, help="Number of top runs to print")
    return parser.parse_args()


def most_impactful_hparam(df: pd.DataFrame) -> tuple[str, float]:
    """Estimate impact by max mean val_accuracy spread across each hyperparameter values."""
    candidate_cols = [
        "optimizer",
        "learning_rate",
        "batch_size",
        "num_layers",
        "activation",
        "loss",
        "weight_init",
        "weight_decay",
        "epochs",
        "hidden_size",
    ]

    impacts = defaultdict(float)
    
    # Create a safe dataframe where lists are converted to strings so pandas can group them
    df_safe = df.copy()
    
    for col in candidate_cols:
        if col not in df_safe.columns:
            continue
            
        # FIX: Convert unhashable types (like lists) to strings
        df_safe[col] = df_safe[col].apply(lambda x: str(x) if isinstance(x, list) else x)
        
        means = df_safe.groupby(col)["val_accuracy"].mean()
        if len(means) > 1:
            impacts[col] = float(means.max() - means.min())

    if not impacts:
        return "unknown", 0.0

    best_col = max(impacts, key=impacts.get)
    return best_col, impacts[best_col]


def main() -> None:
    args = parse_args()

    import wandb

    api = wandb.Api()
    sweep_path = f"{args.entity}/{args.project}/{args.sweep_id}"
    print(f"Fetching sweep data from: {sweep_path} ...")
    sweep = api.sweep(sweep_path)

    rows = []
    for run in sweep.runs:
        # Check if the run finished and has the metric
        summary = run.summary
        if "val_accuracy" not in summary:
            continue
        row = {"run_id": run.id, "run_name": run.name, "val_accuracy": summary["val_accuracy"]}
        
        # Some configs might have lists, keep them as is for printing, stringified in analysis
        row.update(run.config)
        rows.append(row)

    if not rows:
        raise RuntimeError("No completed runs with val_accuracy found in sweep.")

    df = pd.DataFrame(rows)
    df = df.sort_values("val_accuracy", ascending=False)

    best = df.iloc[0]
    hparam, spread = most_impactful_hparam(df)

    print("\n=== 2.2 Sweep Analysis ===")
    print(f"Total completed runs considered: {len(df)}")
    print(f"Most impactful hyperparameter (mean spread heuristic): {hparam} ({spread:.4f})")
    print("\nBest configuration overall:")

    best_cfg_keys = [
        "optimizer",
        "learning_rate",
        "batch_size",
        "num_layers",
        "hidden_size",
        "activation",
        "loss",
        "weight_init",
        "weight_decay",
        "epochs",
    ]
    for key in best_cfg_keys:
        if key in best:
            print(f"  {key}: {best[key]}")
    print(f"  val_accuracy: {best['val_accuracy']:.4f}")

    print(f"\nTop {args.top_k} runs (For Fashion-MNIST Transfer):")
    columns = ["run_name", "val_accuracy", "optimizer", "activation", "num_layers", "hidden_size"]
    columns = [c for c in columns if c in df.columns]
    print(df[columns].head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()