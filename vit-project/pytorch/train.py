"""Training loop for ViT-Tiny / CNN baseline on CIFAR-10 (PyTorch).

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

import torch
import torch.nn as nn

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
        return CNNBaseline(in_channels=3, num_classes=10, base_channels=80)
    raise ValueError(f"unknown model {name}")


def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def run_epoch(model, loader, optimizer, criterion, device, train, lr_schedule=None, step_offset=0):
    model.train(mode=train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    step = step_offset
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train and lr_schedule is not None:
            lr = lr_schedule(step)
            for g in optimizer.param_groups:
                g["lr"] = lr

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_n += images.size(0)
        step += 1

    return total_loss / total_n, total_correct / total_n, step


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
    parser.add_argument("--data-dir", default="../../data")
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = build_model(args.model).to(device)
    n_params = count_parameters(model)
    print(f"model: {args.model}, augment: {args.augment}, params: {n_params:,}")

    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, augment=args.augment,
        num_workers=args.num_workers, subset_size=args.subset_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    lr_schedule = lambda step: cosine_warmup_lr(step, total_steps, warmup_steps, args.lr)

    run_name = args.run_name or f"{args.model}_{'aug' if args.augment else 'plain'}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{run_name}.json"

    history = {"config": vars(args), "params": n_params, "epochs": []}
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, global_step = run_epoch(
            model, train_loader, optimizer, criterion, device,
            train=True, lr_schedule=lr_schedule, step_offset=global_step,
        )
        val_loss, val_acc, _ = run_epoch(
            model, test_loader, optimizer, criterion, device, train=False,
        )
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
