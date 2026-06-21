"""
Mini DeepSeek-V4 Model
======================
Assembles all V4 modules into complete transformer:
- mHC residual connections (not plain add)
- Hybrid CSA+SWA attention
- DeepSeekMoE (some layers) + Dense FFN (other layers)
- Hash routing on first MoE layers
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

from config import DeepSeekV4Config
from modules import (
    RMSNorm, HybridAttention, DeepSeekMoE, DenseFFN,
    ManifoldHyperConnection,
)


class DeepSeekV4Block(nn.Module):
    def __init__(self, config: DeepSeekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        compress_ratio = config.get_compress_ratio(layer_idx)
        self.attention = HybridAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_pos=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            sliding_window=config.sliding_window,
            compress_ratio=compress_ratio,
            dropout=config.dropout,
        )

        is_moe = layer_idx in config.moe_layers
        if is_moe:
            use_hash = layer_idx in config.hash_routing_layers
            self.ffn = DeepSeekMoE(
                hidden_size=config.hidden_size,
                n_routed_experts=config.n_routed_experts,
                n_shared_experts=config.n_shared_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                moe_intermediate_size=config.moe_intermediate_size,
                routed_scaling_factor=config.routed_scaling_factor,
                scoring_func=config.scoring_func,
                norm_topk_prob=config.norm_topk_prob,
                use_hash_routing=use_hash,
            )
        else:
            self.ffn = DenseFFN(config.hidden_size, config.intermediate_size)

        self.attn_hc = ManifoldHyperConnection(
            config.hidden_size, config.hc_mult,
            config.hc_sinkhorn_iters, config.hc_eps,
        )
        self.ffn_hc = ManifoldHyperConnection(
            config.hidden_size, config.hc_mult,
            config.hc_sinkhorn_iters, config.hc_eps,
        )

    def forward(self, x, position_ids):
        attn_input = self.attn_norm(x)
        attn_out = self.attention(attn_input, position_ids)
        x = self.attn_hc(x, attn_out)

        ffn_input = self.ffn_norm(x)
        ffn_out = self.ffn(ffn_input)
        x = self.ffn_hc(x, ffn_out)

        return x


class DeepSeekV4Model(nn.Module):
    def __init__(self, config: DeepSeekV4Config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            DeepSeekV4Block(config, i) for i in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if any(pn.endswith(w) for w in ["o_proj.weight", "down_proj.weight", "proj_out.weight"]):
                torch.nn.init.normal_(p, mean=0.0, std=(config.initializer_range / math.sqrt(2 * config.num_hidden_layers)))

        n_params = sum(p.numel() for p in self.parameters())
        n_params_no_embed = n_params - self.embed_tokens.weight.numel()
        if not config.tie_word_embeddings:
            n_params_no_embed -= self.lm_head.weight.numel()
        print(f"Model params: {n_params:,} total | {n_params_no_embed:,} non-embedding")

        moe_layers = config.moe_layers
        hash_layers = config.hash_routing_layers
        print(f"MoE layers: {moe_layers} | Hash routing layers: {hash_layers}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids, labels=None, use_checkpoint=False):
        batch, seq_len = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            if use_checkpoint and self.training:
                x = checkpoint(layer, x, position_ids, use_reentrant=False)
            else:
                x = layer(x, position_ids)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.contiguous().view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}
