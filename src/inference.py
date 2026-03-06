"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import json
import os
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference for Neural Network")
    # Default arguments (will be overridden by config.json)
    parser.add_argument("-d", "--dataset", type=str, default="mnist")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop") 
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument("-sz", "--hidden_size", type=str, nargs="+", default=["128", "128", "128"])
    parser.add_argument("-a", "--activation", type=str, default="relu")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy")
    parser.add_argument("-wi", "--weight_init", type=str, default="xavier")
    
    parser.add_argument("--model_path", type=str, default="src/best_model.npy")
    parser.add_argument("--config_save_path", type=str, default="src/config.json")
    return parser.parse_args()


def load_model_from_disk(model_path: str, config_path: str, args: Any) -> NeuralNetwork:
    """Load config overrides and trained model from disk."""
    
    # 1. Override CLI args with the actual config used during training
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)
        for key, value in saved_config.items():
            setattr(args, key, value)
    
    # 2. Safely parse hidden_size back into a list of integers
    if isinstance(args.hidden_size, str):
         args.hidden_size = [int(x.strip("[] ")) for x in args.hidden_size.split(",")]
    elif len(args.hidden_size) > 0 and isinstance(args.hidden_size[0], str):
        val = args.hidden_size[0]
        if val.startswith("["):
            args.hidden_size = [int(x) for x in val.replace("[", "").replace("]", "").split(",")]
        else:
            args.hidden_size = [int(x) for x in args.hidden_size]

    # 3. Load weights
    data = np.load(model_path, allow_pickle=True).item()
        
    model = NeuralNetwork(args, input_dim=784, num_classes=10)
    model.set_weights(data)
    return model


def evaluate_model(model: NeuralNetwork, X_test: np.ndarray, y_test_onehot: np.ndarray, y_test_labels: np.ndarray, batch_size: int = 512):
    n = X_test.shape[0]
    all_logits = []
    
    # Batched inference to prevent Memory errors on Gradescope
    for start in range(0, n, batch_size):
        end = start + batch_size
        X_batch = X_test[start:end]
        all_logits.append(model.forward(X_batch))
        
    logits = np.vstack(all_logits)
    model.last_logits = logits
    loss, probs = model.compute_loss_and_output(y_test_onehot)

    # argmax of logits provides the exact same classification as argmax of probabilities
    y_pred_labels = np.argmax(logits, axis=1)

    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_labels, y_pred_labels, average="macro", zero_division=0,
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
    args = parse_arguments()

    # Build and load model dynamically using the config.json
    model = load_model_from_disk(args.model_path, args.config_save_path, args)

    # Load test data
    _, _, _, _, X_test, y_test_onehot, y_test_labels = load_dataset(args.dataset)

    results = evaluate_model(model, X_test, y_test_onehot, y_test_labels)

    print(
        f"Test Loss: {results['loss']:.4f}, "
        f"Accuracy: {results['accuracy']:.4f}, "
        f"F1: {results['f1']:.4f}, "
        f"Precision: {results['precision']:.4f}, "
        f"Recall: {results['recall']:.4f}"
    )

if __name__ == "__main__":
    main()