"""Two-stem quantile Transformer.

    market window (B,W,21) -> market stem (MLP) --+
                                                  +-- concat + LayerNorm per session, + position + stock
    news window   (B,W,24) -> news stem   (MLP) --+   -> Transformer encoder -> last session -> head

The head outputs a median and three positive gaps per side; cumulating the gaps outward gives seven
non-crossing quantiles [q01, q025, q05, q50, q95, q975, q99].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import QUANTILES

N_SIDE = 3                                   # quantile levels on each side of the median


class QuantileTransformer(nn.Module):
    def __init__(self, n_price, n_news, n_stocks, d_model=96, n_layers=3, n_heads=4,
                 stock_dim=16, dropout=0.10, window=20):
        super().__init__()
        h = d_model // 2
        self.price_stem = nn.Sequential(nn.Linear(n_price, h), nn.GELU(), nn.Linear(h, h))
        self.news_stem = nn.Sequential(nn.Linear(n_news, h), nn.GELU(), nn.Linear(h, h))
        self.fuse_norm = nn.LayerNorm(d_model)
        self.stock_emb = nn.Embedding(n_stocks, stock_dim)
        self.stock_proj = nn.Linear(stock_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, 1 + 2 * N_SIDE))

    def forward(self, Xp, Xn, sid):
        z = torch.cat([self.price_stem(Xp), self.news_stem(Xn)], dim=-1)      # (B,W,d)
        z = self.fuse_norm(z) + self.pos + self.stock_proj(self.stock_emb(sid)).unsqueeze(1)
        out = self.head(self.encoder(z)[:, -1])                               # last session
        m = out[:, 0]
        lo_gap = torch.cumsum(F.softplus(out[:, 1:1 + N_SIDE]), dim=-1)       # inner -> outer
        hi_gap = torch.cumsum(F.softplus(out[:, 1 + N_SIDE:]), dim=-1)
        lows = m.unsqueeze(-1) - lo_gap.flip(-1)                              # [q01, q025, q05]
        highs = m.unsqueeze(-1) + hi_gap                                      # [q95, q975, q99]
        return torch.cat([lows, m.unsqueeze(-1), highs], dim=-1)              # (B,7)


def pinball_loss(pred, y):
    """Sum over the seven levels of the mean pinball loss. pred (B,7), y (B,)."""
    loss = 0.0
    for j, q in enumerate(QUANTILES):
        d = y - pred[:, j]
        loss = loss + torch.maximum(q * d, (q - 1) * d).mean()
    return loss
