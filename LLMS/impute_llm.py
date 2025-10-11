# impute_llm.py
import math, random, json, os, numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from transformers import (
    AutoConfig, AutoModel, AutoTokenizer, set_seed, get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType

class TimeSeriesWindowDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,          # [N, T, C] floats
        seq_len: int = 96,
        stride: int = 96,
        mask_ratio: float = 0.25,  # {0.125, 0.25, 0.375, 0.5} used in the paper
        normalize: bool = True,
        seed: int = 42,
    ):
        assert data.ndim == 3
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.normalize = normalize
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.N, self.T, self.C = data.shape

        # simple z-norm per variable over the whole set (swap for instance norm if you prefer)
        if normalize:
            self.mean = data.mean(axis=(0,1), keepdims=True)
            self.std  = data.std(axis=(0,1), keepdims=True) + 1e-8
            data = (data - self.mean) / self.std
        else:
            self.mean = np.zeros((1,1,self.C), dtype=np.float32)
            self.std  = np.ones((1,1,self.C), dtype=np.float32)

        # build windows
        self.windows = []
        for i in range(self.N):
            t = 0
            while t + seq_len <= self.T:
                self.windows.append(data[i, t:t+seq_len, :].astype(np.float32))
                t += stride
        self.windows = np.stack(self.windows, axis=0)  # [M, L, C]
        self.M = self.windows.shape[0]
        self.rng = rng

    def __len__(self):
        return self.M

    def _mcar_mask(self, L, C) -> np.ndarray:
        # independent Bernoulli for each value
        mask = self.rng.random((L, C)) < self.mask_ratio
        return mask.astype(np.bool_)

    def __getitem__(self, idx):
        x = self.windows[idx]                   # [L, C]
        mask = self._mcar_mask(*x.shape)        # [L, C] True where we will compute loss
        x_input = x.copy()
        # Option A: leave inputs as-is and compute loss only on masked
        # Option B: optionally zero-out masked inputs to encourage denoising
        x_input[mask] = 0.0
        return {
            "x_input": torch.from_numpy(x_input),  # model input
            "x_true":  torch.from_numpy(x),        # target
            "mask":    torch.from_numpy(mask),     # boolean mask
        }

class TSLLMImputer(nn.Module):
    def __init__(
        self,
        backbone_name: str = "bert-base-uncased",  # "gpt2", "t5-small", "facebook/bart-base", "meta-llama/Llama-2-7b-hf" (if you have VRAM)
        num_vars: int = 5,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        tune_layers_fraction: float = 0.25,        # train only last 25% transformer blocks (paper’s choice)
        peft_method: str = "lora",                 # "lora" | "qlora" | "adalora"
        seq_len: int = 96,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.seq_len  = seq_len

        self.cfg = AutoConfig.from_pretrained(backbone_name)
        # Use the *base* model to get hidden states (not the LM head):
        self.backbone = AutoModel.from_pretrained(backbone_name, torch_dtype=torch.float32)

        hidden = self.cfg.hidden_size if hasattr(self.cfg, "hidden_size") else self.cfg.d_model
        self.input_proj  = nn.Linear(num_vars, hidden)
        self.pos_embed   = nn.Parameter(torch.zeros(1, seq_len, hidden))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.output_head = nn.Linear(hidden, num_vars)

        # Optionally freeze everything first:
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Unfreeze top-k blocks (25% by default) to mimic the paper’s partial tuning:
        n_blocks = self._count_transformer_blocks()
        k = max(1, int(math.ceil(n_blocks * tune_layers_fraction)))
        self._unfreeze_last_k_blocks(k)

        # Apply PEFT/LoRA to Attention + FFN projections
        target_modules = self._guess_target_modules()
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules, bias="none", task_type=TaskType.FEATURE_EXTRACTION
        )

        # ---- To try AdaLoRA or QLoRA instead: ----
        # from peft import AdaLoraConfig, PromptLearningConfig, get_peft_model
        # lora_cfg = AdaLoraConfig(target_modules=target_modules, init_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        # For QLoRA you’d load a 4-bit quantized backbone via bitsandbytes and then use a standard LoRAConfig.

        self.backbone = get_peft_model(self.backbone, lora_cfg)

    def _count_transformer_blocks(self):
        # heuristics for common models
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return len(self.backbone.encoder.layer)  # BERT, BART-encoder, T5-encoder
        if hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "h"):
            return len(self.backbone.transformer.h)  # GPT-2
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layers"):
            return len(self.backbone.model.layers)   # LLaMA-style
        return 12

    def _unfreeze_last_k_blocks(self, k: int):
        def unfreeze(mod):
            for p in mod.parameters():
                p.requires_grad = True

        # BERT / encoder-like
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            for block in self.backbone.encoder.layer[-k:]:
                unfreeze(block)
        # GPT-2
        if hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "h"):
            for block in self.backbone.transformer.h[-k:]:
                unfreeze(block)
        # LLaMA (transformers>=4.39)
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layers"):
            for block in self.backbone.model.layers[-k:]:
                unfreeze(block)

    def _guess_target_modules(self):
        """
        Choose attention + FFN dense projections across popular architectures.
        The idea is to put LoRA on q/k/v/o and on MLP in/out (FFN),
        as recommended by the paper’s component study. :contentReference[oaicite:4]{index=4}
        """
        return [
            # attention
            "q_proj","k_proj","v_proj","o_proj",          # LLaMA/OPT style
            "query","key","value","dense",                # BERT/BART attention dense
            # FFN / MLP
            "fc_in","fc_out","up_proj","down_proj","gate_proj","intermediate","output"
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C] normalized floats with zeros at masked positions (optional)
        returns: [B, L, C] reconstructed
        """
        B, L, C = x.shape
        h = self.input_proj(x) + self.pos_embed[:, :L, :]
        # Route as inputs_embeds so the backbone can process our continuous embeddings
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone, "get_encoder"):
            # T5/BART encoders
            out = self.backbone(inputs_embeds=h, output_hidden_states=False).last_hidden_state
        else:
            out = self.backbone(inputs_embeds=h, output_hidden_states=False).last_hidden_state
        y = self.output_head(out)  # [B, L, C]
        return y

@dataclass
class TrainConfig:
    backbone: str = "bert-base-uncased"     # try "gpt2" or "t5-small" as well
    seq_len: int = 96
    batch_size: int = 32
    lr: float = 2e-4
    wd: float = 0.0
    epochs: int = 20
    warmup_steps: int = 500
    mask_ratio: float = 0.25
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    tune_layers_fraction: float = 0.25
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

def masked_mse(pred, target, mask):
    # mask == True means "compute loss here"
    m = mask.float()
    diff2 = (pred - target)**2
    num = (diff2 * m).sum()
    den = m.sum().clamp(min=1.0)
    return num / den

def masked_mae(pred, target, mask):
    m = mask.float()
    diff = (pred - target).abs()
    num = (diff * m).sum()
    den = m.sum().clamp(min=1.0)
    return num / den

def train_one_epoch(model, loader, opt, sched, device):
    model.train()
    total_mse = total_mae = 0.0
    steps = 0
    for batch in loader:
        x_in  = batch["x_input"].to(device)   # [B,L,C]
        x_true= batch["x_true"].to(device)
        mask  = batch["mask"].to(device)

        pred = model(x_in)
        loss = masked_mse(pred, x_true, mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched: sched.step()

        with torch.no_grad():
            mae = masked_mae(pred, x_true, mask).item()
            total_mse += loss.item()
            total_mae += mae
            steps += 1
    return total_mse/steps, total_mae/steps

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse = total_mae = 0.0
    steps = 0
    for batch in loader:
        x_in  = batch["x_input"].to(device)
        x_true= batch["x_true"].to(device)
        mask  = batch["mask"].to(device)
        pred = model(x_in)
        total_mse += masked_mse(pred, x_true, mask).item()
        total_mae += masked_mae(pred, x_true, mask).item()
        steps += 1
    return total_mse/steps, total_mae/steps

def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    # ---- Replace this with your real PM10/PM2.5 array: [N, T, C] ----
    # e.g. stack pollutants and/or exogenous vars as channels
    N, T, C = 50, 24*180, 2   # 6 months hourly, 2 vars
    data = np.sin(np.linspace(0, 30, T))[None, :, None] + 0.1*np.random.randn(N, T, C)
    data = data.astype(np.float32)

    train_ds = TimeSeriesWindowDataset(data[:40], seq_len=cfg.seq_len, stride=cfg.seq_len,
                                       mask_ratio=cfg.mask_ratio, normalize=True, seed=cfg.seed)
    val_ds   = TimeSeriesWindowDataset(data[40:], seq_len=cfg.seq_len, stride=cfg.seq_len,
                                       mask_ratio=cfg.mask_ratio, normalize=True, seed=cfg.seed+1)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = TSLLMImputer(
        backbone_name=cfg.backbone,
        num_vars=train_ds.C,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        tune_layers_fraction=cfg.tune_layers_fraction,
        seq_len=cfg.seq_len,
        freeze_backbone=True,
    ).to(cfg.device)

    # Only train adapters + small heads:
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    total_steps = cfg.epochs * len(train_loader)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)

    best_val = float("inf")
    for epoch in range(1, cfg.epochs+1):
        tr_mse, tr_mae = train_one_epoch(model, train_loader, optim, sched, cfg.device)
        va_mse, va_mae = evaluate(model, val_loader, cfg.device)
        print(f"Epoch {epoch:02d} | train MSE {tr_mse:.4f} MAE {tr_mae:.4f} | val MSE {va_mse:.4f} MAE {va_mae:.4f}")
        if va_mse < best_val:
            best_val = va_mse
            torch.save(model.state_dict(), "best_imputer.pt")

if __name__ == "__main__":
    main()
