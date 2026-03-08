"""
inference.py - Load saved model weights and evaluate on test data.

Usage:
    python inference.py --weights best_model.npy --config best_config.json \
                        --dataset mnist

    # Evaluate on custom data (numpy arrays)
    python inference.py --weights best_model.npy --config best_config.json \
                        --dataset mnist --output_file results.json
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from ann.neural_network import MLP
from utils.data_loader import load_dataset



def build_parser():
    parser = argparse.ArgumentParser(description="Inference with saved MLP weights")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to .npy file with saved weights (e.g., best_model.npy)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to best_config.json with model configuration")
    parser.add_argument("-d", "--dataset", type=str, default=None,
                        help="Dataset to evaluate on ('mnist' or 'fashion_mnist'). "
                             "If not given, uses dataset from config.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional: save metrics to this JSON file")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val", "train"],
                        help="Which data split to evaluate on")
    return parser


def run_inference(weights_path: str, config_path: str, dataset_name: str = None,
                  split: str = "test", output_file: str = None):
    """
    Load weights + config, run inference, print and return metrics.

    Returns:
      dict with keys: accuracy, precision, recall, f1_score
    """
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_name = dataset_name or config["dataset"]
    print(f"[Inference] Loading dataset: {dataset_name}, split: {split}")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_dataset(dataset_name)

    split_data = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    X_eval, y_eval = split_data[split]
    print(f"[Inference] Evaluating on {X_eval.shape[0]} samples")

    # ---- Build model ----
    model = MLP(
        input_size=config["input_size"],
        hidden_sizes=config["hidden_sizes"],
        output_size=config["output_size"],
        activation=config["activation"],
        weight_init=config["weight_init"],
    )

    # ---- Load weights ----
    model.load_weights(weights_path)

    # ---- Forward pass (in batches to avoid memory issues) ----
    batch_size = 512
    all_preds = []
    all_probs = []

    for start in range(0, len(X_eval), batch_size):
        X_batch = X_eval[start:start + batch_size]
        logits = model.forward(X_batch)

        # Softmax probabilities
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        preds = np.argmax(probs, axis=1)
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    y_pred = np.array(all_preds)

    # ---- Compute metrics ----
    acc = accuracy_score(y_eval, y_pred)
    prec = precision_score(y_eval, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_eval, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_eval, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_eval, y_pred)

    # ---- Print results ----
    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS ({split.upper()} SET)")
    print("=" * 60)
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("=" * 60)
    print("\nPer-Class Report:")
    print(classification_report(y_eval, y_pred, target_names=class_names, zero_division=0))

    print("\nConfusion Matrix:")
    print(cm)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "dataset": dataset_name,
        "split": split,
        "weights_file": weights_path,
        "config_file": config_path,
    }

    # ---- Save metrics ----
    if output_file:
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"\n[Inference] Results saved to {output_file}")

    return metrics


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    run_inference(
        weights_path=args.weights,
        config_path=args.config,
        dataset_name=args.dataset,
        split=args.split,
        output_file=args.output_file,
    )

# Alias for autograder compatibility
parse_arguments = build_parser
