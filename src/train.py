"""
Main Training Script - Project 1 Optimized
Entry point for training neural networks with robust W&B Sweep support.
"""

import argparse
import json
import os
import ast
from typing import Any, Dict

import numpy as np

# Import NeuralNetwork and data_loader (ensure these paths are correct)
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train/Inference for Neural Network")

    parser.add_argument("-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop", 
                        choices=["sgd", "momentum", "nag", "rmsprop"]) 
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument("-sz", "--hidden_size", type=str, nargs="+", default=["128", "128", "128"])
    parser.add_argument("-a", "--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-l", "--loss", type=str, default="mse", choices=["cross_entropy", "mse"])
    parser.add_argument("-wi", "--weight_init", type=str, default="xavier", choices=["random", "xavier", "zeros"])
    
    parser.add_argument("-w_p", "--wandb_project", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="src/best_model.npy")
    parser.add_argument("--config_save_path", type=str, default="src/config.json")

    return parser.parse_args()

def maybe_init_wandb(args: Any):
    """Initialize W&B if project name provided."""
    if not args.wandb_project:
        return None
    try:
        import wandb
        run = wandb.init(project=args.wandb_project, config=vars(args))
        return run
    except ImportError:
        print("wandb not installed; proceeding without logging.")
        return None

def save_model_and_config(model: NeuralNetwork, args: Any) -> None:
    # Save Configuration
    config = vars(args)
    os.makedirs(os.path.dirname(args.config_save_path) or ".", exist_ok=True)
    with open(args.config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        
    # Save Weights EXACTLY as requested by revised guidelines
    best_weights = model.get_weights()
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    
    # FIX: Wrap in a numpy object array to prevent the inhomogeneous shape error
    weights_array = np.array(best_weights, dtype=object)
    np.save(args.model_path, weights_array, allow_pickle=True)

def main():
    args = parse_arguments()

    # --- 1. Robust Hidden Size Parsing ---
    # W&B often passes lists as strings like "[64, 64]"
    raw_hidden = args.hidden_size
    if len(raw_hidden) == 1 and isinstance(raw_hidden[0], str):
        val = raw_hidden[0]
        if val.startswith("["):
            try:
                args.hidden_size = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                # Fallback: remove brackets and split
                args.hidden_size = [int(x) for x in val.replace('[','').replace(']','').split(',')]
        else:
            args.hidden_size = [int(val)]
    else:
        args.hidden_size = [int(x) for x in raw_hidden]

    # --- 2. Fix Mismatch between num_layers and hidden_size ---
    # This prevents the script from crashing during a Sweep
    actual_h_len = len(args.hidden_size)
    if actual_h_len != args.num_layers:
        print(f"Warning: num_layers ({args.num_layers}) != hidden_size list length ({actual_h_len}).")
        if actual_h_len == 1:
            # Repeat the single value across all layers
            args.hidden_size = args.hidden_size * args.num_layers
        elif actual_h_len > args.num_layers:
            # Truncate
            args.hidden_size = args.hidden_size[:args.num_layers]
        else:
            # Pad with the last value
            padding = [args.hidden_size[-1]] * (args.num_layers - actual_h_len)
            args.hidden_size.extend(padding)
        print(f"Adjusted hidden_size to: {args.hidden_size}")

    # --- 3. Load Data ---
    data = load_dataset(args.dataset)
    # Expected format from your code: (X_train, y_train, X_val, y_val, ...)
    X_train, y_train, X_val, y_val = data[0], data[1], data[2], data[3]

    # --- 4. Initialize W&B ---
    wandb_run = maybe_init_wandb(args)

    # --- 5. Initialize & Train Model ---
    model = NeuralNetwork(args, input_dim=X_train.shape[1], num_classes=y_train.shape[1])

    history = model.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_val,
        y_val=y_val,
        wandb_run=wandb_run,
    )

    # --- 6. Cleanup & Save ---
    if wandb_run is not None:
        import wandb
        # Ensure we log the metric the sweep is looking for: val_accuracy
        final_acc = history.get('val_accuracy', [0])[-1]
        wandb.log({"val_accuracy": final_acc})
        wandb_run.finish()

    save_model_and_config(model, args)
    print(f"Training complete! Model saved to {args.model_path}")

if __name__ == "__main__":
    main()