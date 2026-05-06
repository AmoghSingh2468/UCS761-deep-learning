"""
MedMamba — 2024 medical-specific State Space Model.

Setup:
    git clone https://github.com/YubiaoYue/MedMamba
    pip install mamba-ssm causal-conv1d --no-build-isolation

For RTX 3070 (Ampere sm_86) the standard install works — no special flags needed.
"""
import os
import sys
import torch
import torch.nn as nn

# Absolute path to cloned repo
MEDMAMBA_REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "MedMamba")
)


def build_medmamba(num_classes=1):
    # Only add to sys.path inside this function, and APPEND (not insert)
    # so it never shadows your project's own train.py, config.py, etc.
    if os.path.isdir(MEDMAMBA_REPO) and MEDMAMBA_REPO not in sys.path:
        sys.path.append(MEDMAMBA_REPO)

    try:
        from MedMamba import VSSM as MedMamba
    except ImportError as e:
        raise ImportError(
            f"Failed to import MedMamba. Real error: {e}\n"
            "You likely need: pip install mamba-ssm causal-conv1d --no-build-isolation"
        ) from e

    model = MedMamba(num_classes=num_classes)
    #model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[medmamba] params: {n_params/1e6:.2f}M")

    # Remove MedMamba from sys.path after import so workers don't pick up
    # MedMamba/train.py instead of your project's train.py
    if MEDMAMBA_REPO in sys.path:
        sys.path.remove(MEDMAMBA_REPO)

    return model