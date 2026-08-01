"""Vision Transformer (ViT) built from scratch, plus a CNN baseline of comparable size.

Mirrors pytorch/model.py layer-for-layer so parameter counts and shapes match.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PatchEmbed(layers.Layer):
    """Splits an image into non-overlapping patches and linearly projects each to `embed_dim`.

    A single strided Conv2D does patchify + flatten + linear-project in one op.
    """

    def __init__(self, patch_size=4, embed_dim=192, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)

    def call(self, x):
        # x: (B, H, W, C) -> (B, H/P, W/P, embed_dim) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        B = tf.shape(x)[0]
        D = x.shape[-1]
        x = tf.reshape(x, (B, -1, D))
        return x


class TransformerEncoderBlock(layers.Layer):
    """Pre-norm transformer block: x = x + MHSA(LN(x)); x = x + MLP(LN(x))."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout,
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = keras.Sequential([
            layers.Dense(hidden_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])

    def call(self, x, training=False):
        y = self.norm1(x)
        attn_out = self.attn(y, y, y, training=training)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x), training=training)
        return x


class VisionTransformer(keras.Model):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=192,
        num_layers=6,
        num_heads=3,
        mlp_ratio=2,
        num_classes=10,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(patch_size, embed_dim)

        init = keras.initializers.TruncatedNormal(stddev=0.02)
        self.cls_token = self.add_weight(
            name="cls_token", shape=(1, 1, embed_dim), initializer=init, trainable=True,
        )
        self.pos_embed = self.add_weight(
            name="pos_embed", shape=(1, num_patches + 1, embed_dim), initializer=init, trainable=True,
        )
        self.pos_drop = layers.Dropout(dropout)

        self.blocks = [
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout, name=f"block{i}")
            for i in range(num_layers)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.head = layers.Dense(num_classes, kernel_initializer=init)

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        x = self.patch_embed(x)
        cls_tokens = tf.broadcast_to(self.cls_token, (B, 1, self.cls_token.shape[-1]))
        x = tf.concat([cls_tokens, x], axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


class CNNBaseline(keras.Model):
    """6-layer ResNet-ish CNN baseline, sized to be comparable to ViT-Tiny in param count."""

    def __init__(self, num_classes=10, base_channels=80, **kwargs):
        super().__init__(**kwargs)

        def conv_block(c_out, stride=1):
            return keras.Sequential([
                layers.Conv2D(c_out, kernel_size=3, strides=stride, padding="same", use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
            ])

        c = base_channels
        self.stem = conv_block(c)              # 32x32
        self.layer1 = conv_block(c)             # 32x32
        self.layer2 = conv_block(c * 2, 2)      # 16x16
        self.layer3 = conv_block(c * 2)         # 16x16
        self.layer4 = conv_block(c * 4, 2)      # 8x8
        self.layer5 = conv_block(c * 4)         # 8x8
        self.pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, x, training=False):
        x = self.stem(x, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.layer5(x, training=training)
        x = self.pool(x)
        return self.fc(x)


def count_parameters(model):
    return sum(int(tf.size(w)) for w in model.trainable_weights)


if __name__ == "__main__":
    dummy = tf.random.normal((2, 32, 32, 3))

    patch_embed = PatchEmbed(patch_size=4, embed_dim=192)
    out = patch_embed(dummy)
    print(f"PatchEmbed output shape: {tuple(out.shape)} (expected (2, 64, 192))")
    assert out.shape == (2, 64, 192)

    vit = VisionTransformer()
    logits = vit(dummy)
    print(f"ViT output shape: {tuple(logits.shape)} (expected (2, 10))")
    assert logits.shape == (2, 10)

    cnn = CNNBaseline()
    cnn_logits = cnn(dummy)
    print(f"CNN output shape: {tuple(cnn_logits.shape)} (expected (2, 10))")
    assert cnn_logits.shape == (2, 10)

    print(f"ViT-Tiny params: {count_parameters(vit):,}")
    print(f"CNN baseline params: {count_parameters(cnn):,}")
