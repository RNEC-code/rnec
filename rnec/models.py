import math

import torch
import torch.nn as nn

try:
    from performer_pytorch import Performer
except ImportError as exc:  # pragma: no cover
    Performer = None
    _PERFORMER_IMPORT_ERROR = exc


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :].to(dtype=x.dtype)


class MeanAttentionPooling(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden, max(1, hidden // 2)),
            nn.Tanh(),
            nn.Linear(max(1, hidden // 2), 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(self.score(h), dim=1)
        return torch.sum(attn * h, dim=1)


def _valid_heads(hidden: int, nhead: int) -> tuple[int, int]:
    heads = int(max(1, min(nhead, hidden)))
    while hidden % heads != 0 and heads > 1:
        heads -= 1
    return heads, max(1, hidden // heads)


class PerformerAutoEncoder(nn.Module):
    """Performer encoder-decoder autoencoder for windows shaped [B, C, W]."""

    def __init__(
        self,
        in_channels: int,
        window_size: int,
        hidden: int = 128,
        latent_dim: int = 48,
        dropout: float = 0.1,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int | None = None,
    ):
        super().__init__()
        if Performer is None:
            raise ImportError(
                "The Performer backbone requires performer-pytorch. "
                "Install it with: pip install performer-pytorch"
            ) from _PERFORMER_IMPORT_ERROR

        self.in_channels = int(in_channels)
        self.window_size = int(window_size)
        self.hidden = int(hidden)
        self.latent_dim = int(latent_dim)
        heads, dim_head = _valid_heads(self.hidden, nhead)
        ff_mult = max(1, int(round(float(dim_ff or hidden * 4) / float(hidden))))

        self.in_proj = nn.Linear(in_channels, hidden)
        self.enc_pos = SinusoidalPositionalEncoding(hidden, max_len=max(window_size, 4096))
        self.encoder = Performer(
            dim=hidden,
            depth=int(num_layers),
            heads=heads,
            dim_head=dim_head,
            causal=False,
            ff_mult=ff_mult,
            ff_dropout=dropout,
            attn_dropout=dropout,
        )
        self.enc_norm = nn.LayerNorm(hidden)
        self.pool = MeanAttentionPooling(hidden)
        self.to_latent = nn.Sequential(nn.Linear(hidden, latent_dim), nn.LayerNorm(latent_dim))

        self.latent_to_hidden = nn.Linear(latent_dim, hidden)
        self.dec_pos = nn.Parameter(torch.zeros(1, self.window_size, hidden))
        nn.init.normal_(self.dec_pos, std=0.02)
        self.decoder = Performer(
            dim=hidden,
            depth=int(num_layers),
            heads=heads,
            dim_head=dim_head,
            causal=False,
            ff_mult=ff_mult,
            ff_dropout=dropout,
            attn_dropout=dropout,
        )
        self.dec_norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden, in_channels)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.transpose(1, 2).contiguous()
        h = self.in_proj(tokens)
        h = self.enc_pos(h)
        h = self.encoder(h)
        return self.enc_norm(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.pool(self.encode_tokens(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h0 = self.latent_to_hidden(z).unsqueeze(1)
        tokens = h0.repeat(1, self.window_size, 1) + self.dec_pos[:, : self.window_size, :]
        h = self.decoder(tokens)
        h = self.drop(self.dec_norm(h))
        return self.out_proj(h).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
