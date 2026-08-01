"""Section 6 sanity checks: shape trace, param counts, single-batch overfit."""
import tensorflow as tf
from tensorflow import keras

from model import VisionTransformer, CNNBaseline, count_parameters


def shape_trace():
    print("=== forward pass shape trace (ViT) ===")
    x = tf.random.normal((2, 32, 32, 3))
    print(f"input: {tuple(x.shape)}")

    vit = VisionTransformer()
    patches = vit.patch_embed(x)
    print(f"after patch_embed: {tuple(patches.shape)}")

    B = tf.shape(x)[0]
    cls_tokens = tf.broadcast_to(vit.cls_token, (B, 1, vit.cls_token.shape[-1]))
    tokens = tf.concat([cls_tokens, patches], axis=1)
    print(f"after cls token prepend: {tuple(tokens.shape)}")

    tokens = tokens + vit.pos_embed
    print(f"after pos embed add: {tuple(tokens.shape)}")

    for i, block in enumerate(vit.blocks):
        tokens = block(tokens, training=False)
        print(f"after encoder block {i}: {tuple(tokens.shape)}")

    tokens = vit.norm(tokens)
    cls_out = tokens[:, 0]
    print(f"cls token after final norm: {tuple(cls_out.shape)}")

    logits = vit.head(cls_out)
    print(f"logits: {tuple(logits.shape)}")


def param_counts():
    print("\n=== param counts ===")
    dummy = tf.random.normal((2, 32, 32, 3))
    vit = VisionTransformer()
    cnn = CNNBaseline()
    vit(dummy)
    cnn(dummy)
    print(f"ViT-Tiny: {count_parameters(vit):,}")
    print(f"CNN baseline: {count_parameters(cnn):,}")


def overfit_single_batch(model_name="vit", n_images=16, steps=200):
    print(f"\n=== overfit single batch of {n_images} images ({model_name}) ===")
    tf.random.set_seed(0)
    model = VisionTransformer(dropout=0.0) if model_name == "vit" else CNNBaseline()
    images = tf.random.normal((n_images, 32, 32, 3))
    labels = tf.random.uniform((n_images,), minval=0, maxval=10, dtype=tf.int32)

    optimizer = keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=0.0)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for step in range(steps):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = loss_fn(labels, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 20 == 0 or step == steps - 1:
            preds = tf.argmax(logits, axis=1, output_type=labels.dtype)
            acc = float(tf.reduce_mean(tf.cast(preds == labels, tf.float32)))
            print(f"step {step:4d} | loss {float(loss):.4f} | acc {acc:.3f}")

    final_loss = float(loss)
    assert final_loss < 0.05, f"expected near-zero loss, got {final_loss}"
    print("PASS: model overfits a single batch as expected.")


if __name__ == "__main__":
    shape_trace()
    param_counts()
    overfit_single_batch("vit")
    overfit_single_batch("cnn")
