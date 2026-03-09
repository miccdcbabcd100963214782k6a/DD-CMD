import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.networks.blocks.unetr_block import UnetrUpBlock

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for transformer sequences."""
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation (standard transformer FFN)."""
    def __init__(self, d_model: int, mlp_ratio: float = 2.0, drop: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, d_model),
            nn.Dropout(drop),
        )
    def forward(self, x):
        return self.net(x)

class LocalConvMixer(nn.Module):
    """Lightweight local spatial feature mixing via depthwise-separable convolution."""
    def __init__(self, channels: int, spatial_size: int):
        super().__init__()
        self.c = int(channels)
        self.s = int(spatial_size)
        self.dw = nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, groups=self.c, bias=False)
        self.pw = nn.Conv2d(self.c, self.c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(self.c)
        self.act = nn.GELU()
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        h = w = self.s
        if n != h * w:
            raise ValueError(f"LocalConvMixer expected N={h*w} tokens but got N={n}")
        x = tokens.transpose(1, 2).contiguous().view(b, c, h, w)
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        out = x.flatten(2).transpose(1, 2).contiguous()
        return out

class TextGuidedAttention(nn.Module):
    """Text-guided cross-attention for visual feature enhancement."""
    def __init__(
        self,
        channels: int,
        spatial_size: int,
        out_text_len: int,
        txt_dim: int = 768,
        num_heads: int = 4,
        drop: float = 0.0,
    ):
        super().__init__()
        self.c = int(channels)
        self.s = int(spatial_size)
        self.out_text_len = int(out_text_len)
        self.local = LocalConvMixer(self.c, self.s)
        self.txt_proj = nn.Linear(txt_dim, self.c)
        self.vis_pos = PositionalEncoding(self.c)
        self.txt_pos = PositionalEncoding(self.c, max_len=max(64, self.out_text_len))
        self.norm_q = nn.LayerNorm(self.c)
        self.norm_kv = nn.LayerNorm(self.c)
        self.cross = nn.MultiheadAttention(
            embed_dim=self.c,
            num_heads=num_heads,
            batch_first=True,
            dropout=drop
        )
        self.norm1 = nn.LayerNorm(self.c)
        self.ffn = FeedForward(self.c, mlp_ratio=2.0, drop=drop)
        self.norm2 = nn.LayerNorm(self.c)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.c * 2, max(32, self.c // 2)),
            nn.GELU(),
            nn.Linear(max(32, self.c // 2), 1),
        )
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)
    def _resample_text_tokens(self, txt_tokens: torch.Tensor) -> torch.Tensor:
        """Resample text sequence to fixed length via linear interpolation."""
        b, l, c = txt_tokens.shape
        if l == self.out_text_len:
            return txt_tokens
        x = txt_tokens.transpose(1, 2)
        x = F.interpolate(x, size=self.out_text_len, mode="linear", align_corners=False)
        return x.transpose(1, 2).contiguous()
    def forward(self, vis_tokens: torch.Tensor, txt_tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = vis_tokens.shape
        vis_local = self.local(vis_tokens)
        txt_c = self.txt_proj(txt_tokens)
        txt_c = self._resample_text_tokens(txt_c)
        q = self.vis_pos(self.norm_q(vis_local))
        kv = self.txt_pos(self.norm_kv(txt_c))
        attn_out = self.cross(q, kv, kv)[0]
        vis_pool = vis_local.mean(dim=1)
        txt_pool = txt_c.mean(dim=1)
        g = torch.sigmoid(
            self.gate_mlp(torch.cat([vis_pool, txt_pool], dim=1))
        )
        g = g.view(b, 1, 1)
        x = vis_tokens + self.scale * g * attn_out
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

def _build_dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Construct orthonormal DCT-II basis matrix."""
    i = torch.arange(n, device=device, dtype=dtype).view(1, n)
    k = torch.arange(n, device=device, dtype=dtype).view(n, 1)
    mat = torch.cos(math.pi * (i + 0.5) * k / n)
    mat[0, :] *= 1.0 / math.sqrt(2.0)
    mat = mat * math.sqrt(2.0 / n)
    return mat

class DCT2d(nn.Module):
    """2D Discrete Cosine Transform (DCT-II) for square feature maps."""
    def __init__(self, size: int):
        super().__init__()
        self.size = int(size)
        self.register_buffer("_dct", torch.empty(0), persistent=False)
    def _get_dct(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Lazy initialization of DCT matrix (device/dtype-aware)."""
        if self._dct.numel() == 0 or self._dct.device != device or self._dct.dtype != dtype:
            self._dct = _build_dct_matrix(self.size, device=device, dtype=dtype)
        return self._dct
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h != self.size or w != self.size:
            raise ValueError(f"DCT2d expected {self.size}x{self.size}, got {h}x{w}")
        dct = self._get_dct(x.device, x.dtype)
        x = torch.matmul(dct, x)
        x = torch.matmul(x, dct.t())
        return x

class FrequencyTextModulation(nn.Module):
    """Text-conditioned frequency-domain feature calibration using DCT and FiLM."""
    def __init__(
        self,
        channels: int,
        spatial_size: int,
        txt_dim: int = 768,
        mlp_ratio: float = 0.5
    ):
        super().__init__()
        self.channels = int(channels)
        self.spatial_size = int(spatial_size)
        self.dct2 = DCT2d(self.spatial_size)
        self.freq_gate = nn.Parameter(
            torch.zeros(1, 1, self.spatial_size, self.spatial_size)
        )
        self.txt_proj = nn.Linear(txt_dim, channels)
        hidden = max(32, int((channels * 2) * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels * 2),
        )
        self.norm = nn.LayerNorm(channels)
    def forward(self, tokens: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        h = w = self.spatial_size
        if n != h * w:
            raise ValueError(
                f"Spectral-Text Adaptive Modulation expected N={h*w} tokens but got N={n}"
            )
        x = tokens.transpose(1, 2).contiguous().view(b, c, h, w)
        freq = self.dct2(x)
        gate = torch.sigmoid(self.freq_gate.to(dtype=x.dtype, device=x.device))
        freq_energy = (freq.pow(2) * gate).mean(dim=(2, 3))
        txt_pool = txt.mean(dim=1)
        txt_vec = self.txt_proj(txt_pool)
        cond = torch.cat([freq_energy, txt_vec], dim=1)
        gamma_beta = self.mlp(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = torch.tanh(gamma).view(b, c, 1, 1)
        beta = torch.tanh(beta).view(b, c, 1, 1)
        x = x * (1.0 + gamma) + beta
        out = x.flatten(2).transpose(1, 2).contiguous()
        return self.norm(out)

class TextConditionedDecoder(nn.Module):
    """Text-conditioned decoder block with dual-domain (spatial + frequency) refinement."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_size: int,
        text_len: int
    ) -> None:
        super().__init__()
        self.spatial_size = int(spatial_size)
        self.fuse = TextGuidedAttention(
            channels=in_channels,
            spatial_size=self.spatial_size,
            out_text_len=text_len,
            txt_dim=768,
            num_heads=4,
            drop=0.0,
        )
        self.spectral = FrequencyTextModulation(
            channels=in_channels,
            spatial_size=self.spatial_size,
            txt_dim=768
        )
        self.decoder = UnetrUpBlock(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="BATCH",
        )
    def forward(
        self,
        vis: torch.Tensor,
        skip_vis: torch.Tensor,
        txt: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if txt is not None:
            vis = self.fuse(vis, txt)
            vis = self.spectral(vis, txt)
        vis = rearrange(
            vis,
            "B (H W) C -> B C H W",
            H=self.spatial_size,
            W=self.spatial_size
        )
        skip_vis = rearrange(
            skip_vis,
            "B (H W) C -> B C H W",
            H=self.spatial_size * 2,
            W=self.spatial_size * 2
        )
        out = self.decoder(vis, skip_vis)
        out = rearrange(out, "B C H W -> B (H W) C")
        return out