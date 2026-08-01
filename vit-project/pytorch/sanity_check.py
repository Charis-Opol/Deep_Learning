"""Section 6 sanity checks: shape trace, param counts, single-batch overfit."""
import torch
import torch.nn as nn

from model import VisionTransformer, CNNBaseline, count_parameters


def shape_trace():
    print("=== forward pass shape trace (ViT) ===")
    x = torch.randn(2, 3, 32, 32)
    print(f"input: {tuple(x.shape)}")

    vit = VisionTransformer()
    patches = vit.patch_embed(x)
    print(f"after patch_embed: {tuple(patches.shape)}")

    B = x.shape[0]
    cls_tokens = vit.cls_token.expand(B, -1, -1)
    tokens = torch.cat([cls_tokens, patches], dim=1)
    print(f"after cls token prepend: {tuple(tokens.shape)}")

    tokens = tokens + vit.pos_embed
    print(f"after pos embed add: {tuple(tokens.shape)}")

    for i, block in enumerate(vit.blocks):
        tokens = block(tokens)
        print(f"after encoder block {i}: {tuple(tokens.shape)}")

    tokens = vit.norm(tokens)
    cls_out = tokens[:, 0]
    print(f"cls token after final norm: {tuple(cls_out.shape)}")

    logits = vit.head(cls_out)
    print(f"logits: {tuple(logits.shape)}")


def param_counts():
    print("\n=== param counts ===")
    vit = VisionTransformer()
    cnn = CNNBaseline()
    print(f"ViT-Tiny: {count_parameters(vit):,}")
    print(f"CNN baseline: {count_parameters(cnn):,}")


def overfit_single_batch(model_name="vit", n_images=16, steps=200):
    print(f"\n=== overfit single batch of {n_images} images ({model_name}) ===")
    torch.manual_seed(0)
    model = VisionTransformer(dropout=0.0) if model_name == "vit" else CNNBaseline()
    images = torch.randn(n_images, 3, 32, 32)
    labels = torch.randint(0, 10, (n_images,))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(steps):
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0 or step == steps - 1:
            acc = (logits.argmax(dim=1) == labels).float().mean().item()
            print(f"step {step:4d} | loss {loss.item():.4f} | acc {acc:.3f}")

    final_loss = loss.item()
    assert final_loss < 0.05, f"expected near-zero loss, got {final_loss}"
    print("PASS: model overfits a single batch as expected.")


if __name__ == "__main__":
    shape_trace()
    param_counts()
    overfit_single_batch("vit")
    overfit_single_batch("cnn")
