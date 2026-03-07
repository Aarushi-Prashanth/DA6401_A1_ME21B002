"""
train.py - Main training script for the MLP assignment.

Usage examples:
    # Basic training
    python train.py -d mnist -e 10 -b 32 -l cross_entropy -o adam -lr 0.001 \
                    -nhl 3 -sz 128 128 128 -a relu -w_i xavier

    # With W&B logging
    python train.py -d fashion_mnist -e 20 -b 64 -o nadam -lr 0.001 \
                    -nhl 2 -sz 64 64 -a tanh --wandb

    # Full configuration example
    python train.py --dataset mnist --epochs 15 --batch_size 64 \
                    --loss cross_entropy --optimizer adam --learning_rate 0.001 \
                    --weight_decay 0.0005 --num_layers 3 --hidden_size 128 128 128 \
                    --activation relu --weight_init xavier --wandb
"""

import argparse
import json
import os
import time

import numpy as np

from ann.neural_network import MLP
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer
from utils.data_loader import load_dataset, to_onehot, get_batches, compute_accuracy



def build_parser():
    parser = argparse.ArgumentParser(description="Train MLP on MNIST / Fashion-MNIST")

    parser.add_argument("-d", "--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to use: 'mnist' or 'fashion_mnist'")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Mini-batch size")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse"],
                        help="Loss function to use")
    parser.add_argument("-o", "--optimizer", type=str, default="adam",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        help="Optimizer to use")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0,
                        help="L2 weight decay coefficient")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3,
                        help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", default=[128, 128, 128],
                        help="Hidden layer sizes (list). If one value given, repeated for all layers.")
    parser.add_argument("-a", "--activation", type=str, default="relu",
                        choices=["sigmoid", "tanh", "relu"],
                        help="Activation function for hidden layers")
    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier",
                        choices=["random", "xavier", "zeros"],
                        help="Weight initialization method")

    # Optional flags
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="da6401_assignment1",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity name")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="Directory to save model weights and config")
    parser.add_argument("--log_grad_norms", action="store_true",
                        help="Log gradient norms per layer to W&B")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser


def train(args):
    #Reproducibility
    np.random.seed(args.seed)

    #Parse hidden sizes
    hidden_sizes = args.hidden_size
    if len(hidden_sizes) == 1:
        hidden_sizes = hidden_sizes * args.num_layers
    elif len(hidden_sizes) != args.num_layers:
        print(f"[WARNING] num_layers={args.num_layers} but {len(hidden_sizes)} sizes given. "
              f"Using the provided sizes and ignoring num_layers.")
        args.num_layers = len(hidden_sizes)

    #Load data
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_dataset(args.dataset)
    input_size = X_train.shape[1]  # 784
    num_classes = len(class_names)  # 10

    #Build model
    model = MLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=num_classes,
        activation=args.activation,
        weight_init=args.weight_init,
    )
    print(f"[Model] {model}")

    #Loss and optimizer
    criterion = get_loss(args.loss)
    optimizer_kwargs = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    optimizer = get_optimizer(args.optimizer, **optimizer_kwargs)

    # W&B initialization
    run = None
    if args.wandb:
        try:
            import wandb
            config = vars(args)
            config["hidden_sizes"] = hidden_sizes
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=config,
            )
            print(f"[W&B] Run initialized: {run.name}")
        except ImportError:
            print("[WARNING] wandb not installed. Skipping W&B logging.")
            args.wandb = False

    #Training loop
    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"Training: {args.epochs} epochs, batch={args.batch_size}, "
          f"lr={args.learning_rate}, optimizer={args.optimizer}")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        #Training
        train_loss_sum = 0.0
        train_correct = 0
        num_train = 0

        for X_batch, y_batch in get_batches(X_train, y_train, args.batch_size, shuffle=True):
            y_onehot = to_onehot(y_batch, num_classes)

            # Forward
            logits = model.forward(X_batch)

            # Loss
            batch_loss = criterion.forward(logits, y_onehot)
            train_loss_sum += batch_loss * len(y_batch)

            # Predictions
            preds = np.argmax(criterion.get_probs(), axis=1)
            train_correct += np.sum(preds == y_batch)
            num_train += len(y_batch)

            # Backward
            grad_loss = criterion.backward()
            model.backward(grad_loss)

            # Optimizer step
            model.update_params(optimizer)

        # ---- Validation ----
        val_logits = model.forward(X_val)
        y_val_onehot = to_onehot(y_val, num_classes)
        val_loss = criterion.forward(val_logits, y_val_onehot)
        val_probs = criterion.get_probs()
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = compute_accuracy(val_preds, y_val)

        # ---- Epoch metrics ----
        train_loss = train_loss_sum / num_train
        train_acc = train_correct / num_train

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        #W&B logging 
        if args.wandb and run is not None:
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }

            # Optional: gradient norm logging
            if args.log_grad_norms:
                grad_norms = model.get_gradient_norms()
                for i, norm in enumerate(grad_norms):
                    log_dict[f"grad_norm_layer_{i}"] = norm

            wandb.log(log_dict)

        #Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            model.save_weights(os.path.join(args.save_dir, "best_model.npy"))

    #Final test evaluation
    model.load_weights(os.path.join(args.save_dir, "best_model.npy"))
    test_logits = model.forward(X_test)
    y_test_onehot = to_onehot(y_test, num_classes)
    test_loss = criterion.forward(test_logits, y_test_onehot)
    test_preds = np.argmax(criterion.get_probs(), axis=1)
    test_acc = compute_accuracy(test_preds, y_test)

    print(f"\n{'='*60}")
    print(f"Best Val Acc: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    print(f"{'='*60}\n")

    #Save best config
    best_config = {
        "dataset": args.dataset,
        "input_size": input_size,
        "hidden_sizes": hidden_sizes,
        "output_size": num_classes,
        "activation": args.activation,
        "weight_init": args.weight_init,
        "optimizer": args.optimizer,
        "loss": args.loss,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "best_epoch": best_epoch,
    }
    config_path = os.path.join(args.save_dir, "best_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=4)
    print(f"[Config] Saved to {config_path}")

    # Final W&B logging 
    if args.wandb and run is not None:
        wandb.log({
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "best_val_accuracy": best_val_acc,
        })
        # Save model artifact
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(os.path.join(args.save_dir, "best_model.npy"))
        artifact.add_file(config_path)
        run.log_artifact(artifact)
        run.finish()

    return model, history, best_config


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
