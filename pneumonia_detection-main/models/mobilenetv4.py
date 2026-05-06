"""
MobileNetV4 — Google, ECCV 2024.
Universal Inverted Bottleneck (UIB) CNN architecture.
Paper: arXiv:2404.10518 (April 2024)
Available via timm — no extra dependencies needed.
"""
import timm


def build_mobilenetv4(num_classes=1):
    model = timm.create_model(
        'mobilenetv4_conv_medium.e500_r224_in1k',
        pretrained=True,
        num_classes=num_classes,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[mobilenetv4] params: {n_params/1e6:.2f}M")
    return model