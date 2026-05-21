from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    vocab_size: int = 16384
    seq_len: int = 512
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    n_experts: int = 4
    top_k: int = 2
    expert_hidden: int = 1024


# ---- Normalization ----

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


# ---- Rotary Position Embeddings ----

def precompute_rotary_emb(dim: int, max_seq_len: int, device: torch.device):
    theta = 1.0 / (
        10000 ** (torch.arange(0, dim, 2, device=device).float() / dim)
    )
    positions = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(positions, theta)
    cos = freqs.cos()[None, :, None, :]
    sin = freqs.sin()[None, :, None, :]
    return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack([out1, out2], dim=-1)
    return out.flatten(-2)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, n_kv_heads, T, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(B, n_kv_heads, n_rep, T, head_dim)
    return x.reshape(B, n_kv_heads * n_rep, T, head_dim)


# ---- Grouped-Query Attention ----

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        assert args.dim % args.n_heads == 0

        self.qkv_dim = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(args.dim, self.qkv_dim, bias=False)
        self.out_proj = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x, freqs_cis, start_pos=0, use_cache=False, past_kv=None):
        B, T, C = x.shape
        cos = freqs_cis[0][:, start_pos:start_pos + T, :, :]
        sin = freqs_cis[1][:, start_pos:start_pos + T, :, :]

        qkv = self.qkv_proj(x)
        q, k, v = torch.split(
            qkv,
            [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim],
            dim=-1
        )

        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rotary(q, cos, sin).transpose(1, 2)
        k = apply_rotary(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        n_rep = self.n_heads // self.n_kv_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        is_causal = T > 1
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out), present_kv


# ---- Mixture of Experts ----

class Expert(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.expert_hidden, bias=False)
        self.w2 = nn.Linear(args.dim, args.expert_hidden, bias=False)
        self.w3 = nn.Linear(args.expert_hidden, args.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_experts = args.n_experts
        self.top_k = args.top_k

        self.router = nn.Linear(args.dim, args.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(args) for _ in range(args.n_experts)])

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)
        output = torch.zeros_like(x_flat)

        expert_mask = F.one_hot(topk_idx, num_classes=self.n_experts).permute(2, 0, 1)

        for expert_id in range(self.n_experts):
            token_idx, k_idx = torch.where(expert_mask[expert_id])

            if token_idx.numel() > 0:
                selected_tokens = x_flat[token_idx]
                expert_out = self.experts[expert_id](selected_tokens)

                expert_prob = topk_probs[token_idx, k_idx].unsqueeze(-1)
                expert_out = expert_out * expert_prob

                output.index_add_(0, token_idx, expert_out.to(output.dtype))

        importance = router_probs.mean(dim=0)
        aux_loss = (importance * importance).sum() * self.n_experts

        return output.view(B, T, C), aux_loss


# ---- Transformer ----

class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm1 = RMSNorm(args.dim)
        self.attn = SelfAttention(args)
        self.norm2 = RMSNorm(args.dim)
        self.moe = MoE(args)

    def forward(self, x, freqs_cis, start_pos=0, use_cache=False, past_kv=None):
        attn_out, present_kv = self.attn(
            self.norm1(x), freqs_cis, start_pos, use_cache, past_kv
        )
        x = x + attn_out

        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out

        return x, aux_loss, present_kv


class TinyMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        torch.nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, freqs_cis, targets=None, start_pos=0, use_cache=False, past_kvs=None):
        x = self.embed(input_ids)
        total_aux_loss = 0.0
        present_kvs = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None

            x, aux_loss, present_kv = layer(x, freqs_cis, start_pos, use_cache, past_kv)
            total_aux_loss += aux_loss

            if use_cache:
                present_kvs.append(present_kv)

        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None

        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce_loss + (0.01 * total_aux_loss)

        return logits, loss, present_kvs
