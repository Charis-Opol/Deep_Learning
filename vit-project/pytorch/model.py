"""Vision Transformer (ViT) built from scratch, plus a CNN baseline of comparable size."""
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Splits an image into non-overlapping patches and linearly projects each to `embed_dim`.

    A single strided Conv2d does patchify + flatten + linear-project in one op: the conv
    weights are mathematically identical to "flatten each patch, then apply one Linear layer".
    """

    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer block: x = x + MHSA(LN(x)); x = x + MLP(LN(x))."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
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
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


class CNNBaseline(nn.Module):
    """6-layer ResNet-ish CNN baseline, sized to be comparable to ViT-Tiny in param count."""

    def __init__(self, in_channels=3, num_classes=10, base_channels=80):
        super().__init__()

        def conv_block(c_in, c_out, stride=1):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        c = base_channels
        self.stem = conv_block(in_channels, c)                # 32x32
        self.layer1 = conv_block(c, c)                         # 32x32
        self.layer2 = conv_block(c, c * 2, stride=2)           # 16x16
        self.layer3 = conv_block(c * 2, c * 2)                 # 16x16
        self.layer4 = conv_block(c * 2, c * 4, stride=2)       # 8x8
        self.layer5 = conv_block(c * 4, c * 4)                 # 8x8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Step 1 unit test: PatchEmbed shape check
    patch_embed = PatchEmbed(image_size=32, patch_size=4, in_channels=3, embed_dim=192)
    dummy = torch.randn(2, 3, 32, 32)
    out = patch_embed(dummy)
    print(f"PatchEmbed output shape: {tuple(out.shape)} (expected (2, 64, 192))")
    assert out.shape == (2, 64, 192)

    # Step 2: CLS token + pos embed shape check
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
