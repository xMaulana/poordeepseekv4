"""
DeepSeek-V4 Modules
===================
Core building blocks: RMSNorm, RoPE, mHC, DeepSeekMoE, HybridAttention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * (x / rms)).to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_pos=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].float()
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)



class ManifoldHyperConnection(nn.Module):
    def __init__(self, hidden_size, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.expanded_size = hidden_size * hc_mult

        self.log_alpha = nn.Parameter(torch.zeros(hc_mult, hc_mult))
        self.proj_in = nn.Linear(hidden_size, self.expanded_size, bias=False)
        self.proj_out = nn.Linear(self.expanded_size, hidden_size, bias=False)

    def sinkhorn_knopp(self, log_alpha):
        dtype = log_alpha.dtype
        log_alpha = log_alpha.float()
        for _ in range(self.sinkhorn_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        return torch.exp(log_alpha).to(dtype)

    def forward(self, x, sublayer_output):
        batch, seq_len, _ = x.shape
        mixing = self.sinkhorn_knopp(self.log_alpha)

        x_expanded = self.proj_in(x)
        x_streams = x_expanded.view(batch, seq_len, self.hc_mult, self.hidden_size)
        stream_0 = x_streams[:, :, 0:1, :] + sublayer_output.unsqueeze(2)
        x_streams = torch.cat([stream_0, x_streams[:, :, 1:, :]], dim=2)

        x_mixed = torch.einsum("ij,bsjd->bsid", mixing, x_streams)
        x_flat = x_mixed.reshape(batch, seq_len, self.expanded_size)
        
        return x + sublayer_output + self.proj_out(x_flat)


def sqrt_softplus(x):
    return torch.sqrt(F.softplus(x) + 1e-6)


class ExpertFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMoE(nn.Module):
    def __init__(self, hidden_size, n_routed_experts=8, n_shared_experts=1,
                 num_experts_per_tok=2, moe_intermediate_size=1024,
                 routed_scaling_factor=2.5, scoring_func="sqrtsoftplus",
                 norm_topk_prob=True, use_hash_routing=False):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.use_hash_routing = use_hash_routing

        self.shared_experts = nn.ModuleList([
            ExpertFFN(hidden_size, moe_intermediate_size) for _ in range(n_shared_experts)
        ])
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_size, moe_intermediate_size) for _ in range(n_routed_experts)
        ])
        if not use_hash_routing:
            self.gate = nn.Linear(hidden_size, n_routed_experts, bias=False)

    def hash_route(self, x):
        batch_seq = x.shape[0]
        indices = torch.arange(batch_seq, device=x.device)
        expert_indices = torch.stack([
            (indices + i) % self.n_routed_experts
            for i in range(self.num_experts_per_tok)
        ], dim=-1)
        weights = torch.ones_like(expert_indices, dtype=x.dtype) / self.num_experts_per_tok
        return expert_indices, weights

    def learned_route(self, x):
        logits = self.gate(x)

        if self.num_experts_per_tok == 1:
            scores = torch.sigmoid(logits)
        else:
            scores = sqrt_softplus(logits) if self.scoring_func == "sqrtsoftplus" else torch.sigmoid(logits)
            
        topk_weights, topk_indices = torch.topk(scores, k=self.num_experts_per_tok, dim=-1)

        if self.norm_topk_prob and self.num_experts_per_tok > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-5)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, x):
        batch, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)

        shared_out = sum(expert(x_flat) for expert in self.shared_experts)

        if self.use_hash_routing:
            expert_indices, expert_weights = self.hash_route(x_flat)
        else:
            expert_indices, expert_weights = self.learned_route(x_flat)

        routed_out = torch.zeros_like(x_flat)
        for i in range(self.num_experts_per_tok):
            idx = expert_indices[:, i]
            w = expert_weights[:, i:i+1]
            for eid in range(self.n_routed_experts):
                mask = (idx == eid)
                if mask.any():
                    routed_out[mask] += w[mask] * self.experts[eid](x_flat[mask])

        return (shared_out + routed_out).view(batch, seq_len, hidden)


class DenseFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))



class CompressedKVCache(nn.Module):
    def __init__(self, head_dim, compress_ratio):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.compress_k = nn.Linear(compress_ratio * head_dim, head_dim, bias=False)
        self.compress_v = nn.Linear(compress_ratio * head_dim, head_dim, bias=False)

    def forward(self, k, v):
        batch, num_heads, seq_len, head_dim = k.shape
        r = self.compress_ratio
        usable_len = (seq_len // r) * r
        if usable_len == 0:
            return k, v

        compressed_len = usable_len // r
        k_g = k[:, :, :usable_len].reshape(batch, num_heads, compressed_len, r * head_dim)
        v_g = v[:, :, :usable_len].reshape(batch, num_heads, compressed_len, r * head_dim)
        return self.compress_k(k_g), self.compress_v(v_g)


class HybridAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads,
                 head_dim, max_pos=512, rope_theta=10000.0,
                 sliding_window=64, compress_ratio=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.sliding_window = sliding_window
        self.compress_ratio = compress_ratio
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(head_dim, max_pos, rope_theta)
        self.kv_compress = CompressedKVCache(head_dim, compress_ratio)
        self.attn_dropout = nn.Dropout(dropout)

    def _repeat_kv(self, x):
        if self.num_kv_groups == 1:
            return x
        b, h, s, d = x.shape
        return x.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(b, self.num_heads, s, d)

    def _swa_mask(self, seq_len, device):
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
        window_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(
            diagonal=-(self.sliding_window - 1))
        return mask & window_mask

    def forward(self, x, position_ids):
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(x, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k_swa = self._repeat_kv(k)
        v_swa = self._repeat_kv(v)
        swa_scores = torch.matmul(q, k_swa.transpose(-2, -1)) * self.scale
        swa_mask = self._swa_mask(seq_len, x.device)
        swa_scores = swa_scores.masked_fill(~swa_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        swa_out = torch.matmul(self.attn_dropout(F.softmax(swa_scores, dim=-1).to(v_swa.dtype)), v_swa)

        k_comp, v_comp = self.kv_compress(k, v)
        k_comp = self._repeat_kv(k_comp)
        v_comp = self._repeat_kv(v_comp)
        comp_len = k_comp.shape[2]
        csa_scores = torch.matmul(q, k_comp.transpose(-2, -1)) * self.scale
        i_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)
        j_idx = torch.arange(comp_len, device=x.device).unsqueeze(0)
        max_idx = (i_idx - self.compress_ratio) // self.compress_ratio
        causal = j_idx <= max_idx
        csa_scores = csa_scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        csa_probs = F.softmax(csa_scores, dim=-1)

        csa_probs = torch.nan_to_num(csa_probs, nan=0.0).to(v_comp.dtype)
        csa_out = torch.matmul(self.attn_dropout(csa_probs), v_comp)

        attn_out = (swa_out + csa_out).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_out)
