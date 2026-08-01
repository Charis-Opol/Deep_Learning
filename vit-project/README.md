# Vision Transformer (ViT) from Scratch — PyTorch vs. TensorFlow

Implementation of "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020), built from
first principles (no `timm`, no pretrained weights) in both PyTorch and TensorFlow, trained
on CIFAR-10, and compared against a parameter-matched CNN baseline.

## Layout

```
vit-project/
  pytorch/
    model.py        PatchEmbed, TransformerEncoderBlock, VisionTransformer, CNNBaseline
    dataset.py       CIFAR-10 loaders (plain / augmented) via torchvision
    train.py         AdamW + cosine-warmup training loop
    sanity_check.py  shape trace, param counts, single-batch overfit test
  tensorflow/
    model.py         mirrors pytorch/model.py layer-for-layer
    dataset.py        CIFAR-10 loader reading the local pickle batches directly
    train.py          mirrors pytorch/train.py
    sanity_check.py   mirrors pytorch/sanity_check.py
  README.md
```

Both frameworks read CIFAR-10 from the same on-disk batches at `../../data/cifar-10-batches-py`
(the standard Python-pickle format torchvision downloads) — no `tensorflow_datasets` dependency
needed; TensorFlow's loader unpickles those files directly instead of a separate keras.datasets
download.

## Config (identical across both frameworks)

| Param | Value |
|---|---|
| image_size | 32 |
| patch_size | 4 |
| num_patches | 64 |
| embed_dim | 192 |
| num_layers | 6 |
| num_heads | 3 |
| mlp_ratio | 2 |
| num_classes | 10 |
| dropout | 0.1 |

Norm placement is **pre-norm**: `x = x + MHSA(LN(x))`, `x = x + MLP(LN(x))`, MLP is
`Linear(D→4D·mlp_ratio/4) → GELU → Dropout → Linear(→D) → Dropout`. CLS token and position
embeddings are learnable, initialized with truncated-normal `std=0.02` — no sinusoidal encoding.

## Parameter counts

Verified identical between frameworks (`sanity_check.py` in each dir):

| Model | Params |
|---|---|
| ViT-Tiny | 1,806,538 |
| CNN baseline (6-layer, ResNet-ish, base_channels=80) | 1,793,210 |

## Sanity checks (Section 6)

Run in both `pytorch/` and `tensorflow/`:

```
python sanity_check.py
```

Confirms: forward-pass shapes match at every stage, param counts match cross-framework, and
both ViT and CNN overfit a 16-image random batch to near-zero loss (proves model/loss/optimizer
wiring is correct before spending time on a full run).

## Training

```
python train.py --model vit --epochs 50                  # ViT, no augmentation
python train.py --model vit --augment --epochs 50         # ViT + augmentation
python train.py --model cnn --epochs 50                   # CNN baseline
```

Hyperparameters: AdamW (`lr=3e-4`, `weight_decay=0.05`), cosine decay with 5-epoch linear
warmup, batch size 128, label smoothing 0.1. Per-epoch train/val loss and accuracy are logged
to `runs/<run_name>.json` in each framework directory.

**Note on hardware:** this environment has no GPU (`torch.cuda.is_available()` is `False`);
TensorFlow also runs CPU-only. A full 50-epoch run of all three configs in both frameworks is
several hours of CPU time — use `--subset-size` and low `--epochs` for smoke tests, and plan
full runs accordingly (background/overnight, or move to a GPU machine).

## Status

Verified working end-to-end on this machine (CPU-only, no GPU):
- Both `sanity_check.py` scripts pass: shapes match at every stage, param counts match
  cross-framework exactly (1,806,538 / 1,793,210), both models overfit a 16-image batch.
- Both `train.py` scripts run a full train/val epoch loop without error (smoke-tested on a
  512-image subset). No full training run has been executed yet — that's the next step.

One cross-framework wrinkle: Keras 3's `SparseCategoricalCrossentropy` dropped the
`label_smoothing` argument, so `tensorflow/train.py` one-hot-encodes labels and uses
`CategoricalCrossentropy(from_logits=True, label_smoothing=...)` instead — functionally
identical to the PyTorch side's `nn.CrossEntropyLoss(label_smoothing=...)`.

## Results

_To fill in after full training runs complete:_

| Config | Test acc | Params | Epochs |
|---|---|---|---|
| CNN baseline | | 1,793,210 | |
| ViT-Tiny (plain) | | 1,806,538 | |
| ViT-Tiny (augmented) | | 1,806,538 | |

## Expected finding

The CNN baseline should match or beat plain ViT-Tiny on CIFAR-10 from scratch, especially
early in training — its convolutional locality/translation-equivariance prior is a free lunch
at this data scale, exactly as the ViT paper reports. Augmentation should narrow the gap by
partially substituting for the inductive bias ViT lacks. This setup (CIFAR-10, single GPU/CPU
scale) is not expected to reproduce the paper's "ViT wins at scale" crossover — that requires
JFT-300M-scale pretraining data and compute far beyond this project's scope.

## Optional extensions (not yet implemented)

- Attention map visualization (rollout heatmap from the last encoder block's CLS attention)
- Position embedding interpolation for fine-tuning at a different resolution
- Hybrid variant: CNN stem before the patch-embedding projection
- DeiT-style distillation token
