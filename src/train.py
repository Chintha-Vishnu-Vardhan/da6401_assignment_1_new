"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import os
import ast
from typing import Any, Dict

import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():
    """
    Parse command-line arguments.

    Arguments (aligned with assignment spec):
    -d, --dataset: 'mnist' or 'fashion'
    -e, --epochs: Number of training epochs
    -b, --batch_size: Mini-batch size
    -o, --optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    -lr, --learning_rate: Learning rate for optimizer
    -wd, --weight_decay: L2 regularization strength
    -nhl, --num_layers: Number of hidden layers
    -sz, --hidden_size: Hidden layer sizes (provide multiple values)
    -a, --activation: Activation function ('relu', 'sigmoid', 'tanh')
    -l, --loss: Loss function ('cross_entropy', 'mse')
    -wi, --weight_init: Weight initialization method ('random', 'xavier', 'zeros')
    --wandb_project: W&B project name (optional)
    --model_save_path: Relative path to save trained model (.npy)
    --config_save_path: Relative path to save config (.json)
    """
    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=["mnist", "fashion", "fashion_mnist"],
        help="Dataset to use",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
        help="Optimization algorithm",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=0.0,
        help="L2 weight decay (regularization strength)",
    )
    parser.add_argument(
        "-nhl",
        "--num_layers",
        type=int,
        default=2,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "-sz",
        "--hidden_size",
        type=str,
        nargs="+",
        default=[128, 128],
        help="List of hidden layer sizes",
    )
    parser.add_argument(
        "-a",
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "sigmoid", "tanh"],
        help="Hidden layer activation function",
    )
    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "mse"],
        help="Loss function",
    )
    parser.add_argument(
        "-wi",
        "--weight_init",
        type=str,
        default="xavier",
        choices=["random", "xavier", "zeros"],
        help="Weight initialization method",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name (optional)",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="best_model.npy",
        help="Relative path to save trained model weights (.npy)",
    )
    parser.add_argument(
        "--config_save_path",
        type=str,
        default="config.json",
        help="Relative path to save model configuration (.json)",
    )

    return parser.parse_args()


def maybe_init_wandb(args: Any):
    """Initialize W&B if a project name is provided and wandb is installed."""
    if not args.wandb_project:
        return None
    try:
        import wandb

        run = wandb.init(project=args.wandb_project, config=vars(args))
        return run
    except ImportError:
        print("wandb not installed; proceeding without W&B logging.")
        return None


def save_model_and_config(
    model: NeuralNetwork,
    args: Any,
    history: Dict[str, Any],
) -> None:
    """Save model weights to .npy and configuration to .json."""
    # Aggregate weights and biases
    layers_params = []
    for layer in model.layers:
        layers_params.append(
            {
                "W": layer.W,
                "b": layer.b,
                "input_dim": layer.input_dim,
                "output_dim": layer.output_dim,
                "activation": layer.activation_name,
            }
        )

    model_dict = {
        "layers": layers_params,
        "input_dim": model.input_dim,
        "num_classes": model.num_classes,
        "activation": model.activation_name,
        "loss": model.loss_name,
        "history": history,
    }

    model_save_path = args.model_save_path
    config_save_path = args.config_save_path

    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(config_save_path) or ".", exist_ok=True)

    # Save weights
    np.save(model_save_path, model_dict, allow_pickle=True)

    # Save config (JSON-serializable)
    config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "activation": args.activation,
        "loss": args.loss,
        "weight_init": args.weight_init,
        "model_path": model_save_path,
    }

    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def main():
    """
    Main training function.
    """
    args = parse_arguments()

    # Convert hidden_size arguments (strings) into a list of ints.
    # Supports both:
    #   - CLI style: -sz 128 128
    #   - wandb style: --hidden_size="[128, 128]"
    raw_hidden = args.hidden_size
    if len(raw_hidden) == 1 and isinstance(raw_hidden[0], str) and raw_hidden[0].startswith("["):
        try:
            parsed_list = ast.literal_eval(raw_hidden[0])
            args.hidden_size = [int(x) for x in parsed_list]
        except (SyntaxError, ValueError, TypeError):
            raise ValueError(f"Could not parse hidden_size list from {raw_hidden[0]!r}")
    else:
        args.hidden_size = [int(x) for x in raw_hidden]

    # Ensure hidden_size length matches num_layers
    if len(args.hidden_size) == 1 and args.num_layers > 1:
        args.hidden_size = args.hidden_size * args.num_layers
    elif len(args.hidden_size) != args.num_layers:
        # Fallback: truncate or pad with last value
        if len(args.hidden_size) > args.num_layers:
            args.hidden_size = args.hidden_size[: args.num_layers]
        else:
            last = args.hidden_size[-1]
            args.hidden_size = args.hidden_size + [last] * (args.num_layers - len(args.hidden_size))

    # Load data
    (
        X_train,
        y_train,
        X_val,
        y_val,
        _X_test,
        _y_test_onehot,
        _y_test_labels,
    ) = load_dataset(args.dataset)

    # Initialize model
    model = NeuralNetwork(args, input_dim=X_train.shape[1], num_classes=y_train.shape[1])

    # Optional W&B
    wandb_run = maybe_init_wandb(args)

    history = model.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_val,
        y_val=y_val,
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        wandb_run.finish()

    # Save trained model and config
    save_model_and_config(model, args, history)

    print("Training complete!")


if __name__ == "__main__":
    main()

