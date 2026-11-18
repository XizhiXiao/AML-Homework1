import math
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    context_length: int = 1024
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    d_ff: int = 4096
    dropout: float = 0.0
    tie_weights: bool = True

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.d_model, self.n_layers, self.n_heads, self.d_ff) <= 0:
            raise ValueError('All dimensions must be positive')
        if self.d_model % self.n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')


class CausalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.head_dim = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer('mask', torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool)), persistent=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.reshape(b, t, self.heads, self.head_dim).transpose(1, 2) for z in (q, k, v)]
        # Explicit implementation makes masking and attention inspectable.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:t, :t], float('-inf'))
        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out = self.dropout(weights) @ v
        return self.dropout(self.proj(out.transpose(1, 2).contiguous().reshape(b, t, c)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = CausalAttention(cfg)
        self.ff = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
                                nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.token_embedding.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith('attn.proj.weight') or name.endswith('ff.2.weight'):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        embeddings = self.token_embedding.weight.numel() + self.position_embedding.weight.numel()
        return {'total': total, 'non_embedding': total - embeddings}

    def forward(self, ids, targets=None):
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.config.context_length:
            raise ValueError('Input must be [batch, length] within context_length')
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.dropout(self.token_embedding(ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.final_norm(x))
        loss = None if targets is None else F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=100, temperature=0.8, top_k=50, eos_id=None):
        if temperature < 0 or top_k < 0 or max_new_tokens < 0:
            raise ValueError('Sampling arguments must be nonnegative')
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.config.context_length:])[0][:, -1].float()
            if temperature == 0:
                token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float('-inf'))
                token = torch.multinomial(logits.softmax(dim=-1), 1)
            ids = torch.cat([ids, token], dim=1)
            if eos_id is not None and (token == eos_id).all():
                break
        return ids
