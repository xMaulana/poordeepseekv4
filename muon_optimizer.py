import torch
from torch.optim import Muon

def split_params_for_optimizer(model, muon_lr=0.02, adamw_lr=3e-4, weight_decay=0.01):
    muon_params = []
    adamw_params = []
    adamw_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1:
            adamw_no_decay.append(param)
            continue
        if "embed" in name or "lm_head" in name:
            adamw_params.append(param)
            continue
        muon_params.append(param)

    optimizer_groups = {
        "muon": Muon(
            [{"params": muon_params, "lr": muon_lr}],
            lr=muon_lr,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            weight_decay=weight_decay,
        ),
        "adamw": torch.optim.AdamW(
            [
                {"params": adamw_params, "lr": adamw_lr, "weight_decay": weight_decay},
                {"params": adamw_no_decay, "lr": adamw_lr, "weight_decay": 0.0},
            ],
            lr=adamw_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        ),
    }

    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_params) + sum(p.numel() for p in adamw_no_decay)
    print(f"Muon params: {n_muon:,} | AdamW params: {n_adamw:,}")

    return optimizer_groups
