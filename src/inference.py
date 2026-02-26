"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import json
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():
    """
    Parse command-line arguments for inference.

    Arguments:
    - model_path: Path to saved model weights (relative path)
    - config_path: Path to saved config.json (optional but recommended)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    """
    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Relative path to saved model weights (.npy)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.json",
        help="Relative path to model configuration (.json)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=["mnist", "fashion", "fashion_mnist"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for inference",
    )

    return parser.parse_args()


def load_model(model_path: str, config: Dict[str, Any]) -> NeuralNetwork:
    """
    Load trained model from disk.
    """
    # Create dummy args namespace mimicking training args
    class DummyArgs:
        pass

    args = DummyArgs()
    for k, v in config.items():
        setattr(args, k, v)

    # Initialize a fresh network with same architecture
    model = NeuralNetwork(
        args,
        input_dim=config.get("input_dim", 784),
        num_classes=config.get("num_classes", 10),
    )

    # Load saved parameters
    saved = np.load(model_path, allow_pickle=True).item()
    for layer, layer_params in zip(model.layers, saved["layers"]):
        layer.W = layer_params["W"]
        layer.b = layer_params["b"]

    return model


def evaluate_model(model: NeuralNetwork, X_test: np.ndarray, y_test_onehot: np.ndarray, y_test_labels: np.ndarray):
    """
    Evaluate model on test data.

    Returns:
        Dictionary with keys: logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test)
    loss, _ = model.compute_loss_and_output(y_test_onehot)

    y_pred_labels = np.argmax(model.last_probs if model.last_probs is not None else logits, axis=1)

    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_labels,
        y_pred_labels,
        average="macro",
        zero_division=0,
    )

    return {
        "logits": logits,
        "loss": float(loss),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def main():
    """
    Main inference function.

    Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()

    # Load config if available
    try:
        with open(args.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        # Minimal fallback config; assumes default training args were used
        config = {
            "dataset": args.dataset,
            "epochs": 0,
            "batch_size": args.batch_size,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "num_layers": 2,
            "hidden_size": [128, 128],
            "activation": "relu",
            "loss": "cross_entropy",
            "weight_init": "xavier",
        }

    # Ensure dataset consistency
    config["dataset"] = args.dataset

    # Load data (test split only)
    (
        _X_train,
        _y_train,
        _X_val,
        _y_val,
        X_test,
        y_test_onehot,
        y_test_labels,
    ) = load_dataset(args.dataset)

    # Build and load model
    model = load_model(args.model_path, config)

    results = evaluate_model(model, X_test, y_test_onehot, y_test_labels)

    print(
        f"Test Loss: {results['loss']:.4f}, "
        f"Accuracy: {results['accuracy']:.4f}, "
        f"F1: {results['f1']:.4f}, "
        f"Precision: {results['precision']:.4f}, "
        f"Recall: {results['recall']:.4f}"
    )

    return results


if __name__ == "__main__":
    main()

