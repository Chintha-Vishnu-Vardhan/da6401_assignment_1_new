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


def load_model_from_disk(model_path: str, args: Any) -> NeuralNetwork:
    """
    Load trained model from disk as per revised guidelines.
    """
    data = np.load(model_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        weights = data.item()
    else:
        weights = list(data)
        
    model = NeuralNetwork(args, input_dim=784, num_classes=10)
    model.set_weights(weights)
    return model


def evaluate_model(model: NeuralNetwork, X_test: np.ndarray, y_test_onehot: np.ndarray, y_test_labels: np.ndarray):
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
    args = parse_arguments()

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

    # Build and load model using the updated function
    model = load_model_from_disk(args.model_path, args)

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