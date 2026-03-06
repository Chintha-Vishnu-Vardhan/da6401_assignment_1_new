"""
Main Training Script
Entry point for training neural networks with W&B Sweep support.
"""

import argparse
import ast
import json
import os
from typing import Any, Dict

import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Neural Network")

    parser.add_argument("-d",   "--dataset",       type=str,   default="mnist",
                        choices=["mnist", "fashion", "fashion_mnist"])
    parser.add_argument("-e",   "--epochs",         type=int,   default=10)
    parser.add_argument("-b",   "--batch_size",     type=int,   default=128)
    parser.add_argument("-o",   "--optimizer",      type=str,   default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr",  "--learning_rate",  type=float, default=0.001)
    parser.add_argument("-wd",  "--weight_decay",   type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers",     type=int,   default=3)
    parser.add_argument("-sz",  "--hidden_size",    type=str,   nargs="+",
                        default=["128", "128", "128"])
    parser.add_argument("-a",   "--activation",     type=str,   default="relu",
                        choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-l",   "--loss",           type=str,   default="cross_entropy",
                        choices=["cross_entropy", "mse"])
    parser.add_argument("-wi",  "--weight_init",    type=str,   default="xavier",
                        choices=["random", "xavier", "zeros"])
    parser.add_argument("-w_p", "--wandb_project",  type=str,   default=None)
    parser.add_argument("--model_path",             type=str,   default="src/best_model.npy")
    parser.add_argument("--config_save_path",       type=str,   default="src/config.json")

    return parser.parse_args()


def maybe_init_wandb(args: Any):
    if not args.wandb_project:
        return None
    try:
        import wandb
        return wandb.init(project=args.wandb_project, config=vars(args))
    except ImportError:
        print("wandb not installed; proceeding without logging.")
        return None


def save_model_and_config(model: NeuralNetwork, args: Any) -> None:
    # ── Config ──────────────────────────────────────────────────────────────
    config = vars(args)
    os.makedirs(os.path.dirname(args.config_save_path) or ".", exist_ok=True)
    with open(args.config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # ── Weights ─────────────────────────────────────────────────────────────
    # Save as a 0-d object array so that np.load(...).item() returns the
    # original list-of-tuples, and list(np.load(...)) also works via set_weights.
    best_weights = model.get_weights()          # list of (W, b) tuples
    container = np.empty((), dtype=object)      # 0-d object array
    container[()] = best_weights
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    np.save(args.model_path, container)


def main():
    args = parse_arguments()

    # ── 1. Parse hidden_size ────────────────────────────────────────────────
    raw_hidden = args.hidden_size
    if len(raw_hidden) == 1 and isinstance(raw_hidden[0], str):
        val = raw_hidden[0]
        if val.startswith("["):
            try:
                args.hidden_size = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                args.hidden_size = [
                    int(x)
                    for x in val.replace("[", "").replace("]", "").split(",")
                ]
        else:
            args.hidden_size = [int(val)]
    else:
        args.hidden_size = [int(x) for x in raw_hidden]

    # ── 2. Reconcile num_layers vs hidden_size length ───────────────────────
    actual_h_len = len(args.hidden_size)
    if actual_h_len != args.num_layers:
        print(
            f"Warning: num_layers ({args.num_layers}) != "
            f"hidden_size length ({actual_h_len}). Adjusting."
        )
        if actual_h_len == 1:
            args.hidden_size = args.hidden_size * args.num_layers
        elif actual_h_len > args.num_layers:
            args.hidden_size = args.hidden_size[: args.num_layers]
        else:
            padding = [args.hidden_size[-1]] * (args.num_layers - actual_h_len)
            args.hidden_size.extend(padding)
        print(f"Adjusted hidden_size to: {args.hidden_size}")

    # ── 3. Load Data ────────────────────────────────────────────────────────
    data = load_dataset(args.dataset)
    X_train, y_train, X_val, y_val = data[0], data[1], data[2], data[3]

    # ── 4. W&B ──────────────────────────────────────────────────────────────
    wandb_run = maybe_init_wandb(args)

    # ── 5. Train ────────────────────────────────────────────────────────────
    model = NeuralNetwork(
        args,
        input_dim=X_train.shape[1],
        num_classes=y_train.shape[1],
    )

    history = model.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_val,
        y_val=y_val,
        wandb_run=wandb_run,
    )

    # ── 6. Finalise ─────────────────────────────────────────────────────────
    if wandb_run is not None:
        import wandb
        final_acc = history.get("val_accuracy", [0])[-1]
        wandb.log({"val_accuracy": final_acc})
        wandb_run.finish()

    save_model_and_config(model, args)
    print(f"Training complete. Model saved to {args.model_path}")


if __name__ == "__main__":
    main()