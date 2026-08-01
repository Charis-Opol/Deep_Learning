"""Training loop for ViT-Tiny / CNN baseline on CIFAR-10 (TensorFlow).

Usage:
    python train.py --model vit --augment --epochs 50
    python train.py --model cnn --epochs 50
    python train.py --model vit --epochs 1 --subset-size 512   # quick smoke test
"""
import argparse
import json
import math
import time
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from dataset import get_dataloaders
from model import VisionTransformer, CNNBaseline, count_parameters


def build_model(name):
    if name == "vit":
        return VisionTransformer(
            image_size=32, patch_size=4, in_channels=3,
            embed_dim=192, num_layers=6, num_heads=3, mlp_ratio=2,
            num_classes=10, dropout=0.1,
        )
    elif name == "cnn":
        return CNNBaseline(num_classes=10, base_channels=80)
    raise ValueError(f"unknown model {name}")


class CosineWarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / tf.maximum(1.0, self.total_steps - self.warmup_steps)
        cosine_lr = 0.5 * self.base_lr * (1 + tf.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {"base_lr": self.base_lr, "total_steps": self.total_steps, "warmup_steps": self.warmup_steps}


def run_epoch(model, dataset, optimizer, loss_fn, train, num_classes=10):
    total_loss, total_correct, total_n = 0.0, 0, 0
    for images, labels in dataset:
        labels_onehot = tf.one_hot(labels, num_classes)
        if train:
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = loss_fn(labels_onehot, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        else:
            logits = model(images, training=False)
            loss = loss_fn(labels_onehot, logits)

        bs = images.shape[0]
        total_loss += float(loss) * bs
        preds = tf.argmax(logits, axis=1, output_type=labels.dtype)
        total_correct += int(tf.reduce_sum(tf.cast(preds == labels, tf.int32)))
        total_n += bs

    return total_loss / total_n, total_correct / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vit", "cnn"], required=True)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--data-dir", default="../../data/cifar-10-batches-py")
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    model = build_model(args.model)
    train_ds, test_ds, steps_per_epoch = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, augment=args.augment,
        subset_size=args.subset_size,
    )

    # build the model (creates weights) before counting params
    for images, _ in train_ds.take(1):
        model(images, training=False)
    n_params = count_parameters(model)
    print(f"model: {args.model}, augment: {args.augment}, params: {n_params:,}")

    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    lr_schedule = CosineWarmupSchedule(args.lr, float(total_steps), float(warmup_steps))

    optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=args.weight_decay)
    loss_fn = keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=args.label_smoothing,
    )

    run_name = args.run_name or f"{args.model}_{'aug' if args.augment else 'plain'}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{run_name}.json"

    history = {"config": vars(args), "params": n_params, "epochs": []}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_ds, optimizer, loss_fn, train=True)
        val_loss, val_acc = run_epoch(model, test_ds, optimizer, loss_fn, train=False)
        dt = time.time() - t0
        print(
            f"epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | {dt:.1f}s"
        )
        history["epochs"].append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "seconds": dt,
        })
        log_path.write_text(json.dumps(history, indent=2))

    print(f"done. history saved to {log_path}")


if __name__ == "__main__":
    main()
