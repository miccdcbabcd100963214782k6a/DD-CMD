import os
import torch
import torch.nn as nn
from einops import rearrange, repeat
from monai.networks.blocks.dynunet_block import UnetOutBlock
from transformers import AutoModel
from net.decoder import TextConditionedDecoder
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

class FrozenTextEncoder(nn.Module):
    """Frozen BERT-based text encoder."""
    def __init__(self, bert_type: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            bert_type,
            output_hidden_states=True,
            trust_remote_code=True
        )
        for p in self.model.parameters():
            p.requires_grad = False
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return {"feature": output["hidden_states"]}

class VisionEncoder(nn.Module):
    """Vision encoder that extracts hierarchical visual features."""
    def __init__(self, vision_type: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            vision_type,
            output_hidden_states=True
        )
    def forward(self, x: torch.Tensor):
        output = self.model(x, output_hidden_states=True, return_dict=True)
        return {"feature": output["hidden_states"]}

class LightweightTextModulation(nn.Module):
    """Lightweight text-conditioned FiLM modulation for high-resolution refinement."""
    def __init__(self, channels: int, txt_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(txt_dim, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels * 2),
        )
    def forward(self, x: torch.Tensor, txt_tokens: torch.Tensor) -> torch.Tensor:
        t = txt_tokens.mean(dim=1)
        g = self.proj(t)
        gamma, beta = g.chunk(2, dim=1)
        gamma = torch.tanh(gamma).view(x.size(0), x.size(1), 1, 1)
        beta = torch.tanh(beta).view(x.size(0), x.size(1), 1, 1)
        return x * (1.0 + gamma) + beta

class HighResRefinementBlock(nn.Module):
    """High-resolution refinement block with text conditioning."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, txt_dim: int = 768):
        super().__init__()
        self.ups = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.film = LightweightTextModulation(out_ch, txt_dim=txt_dim)
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        txt_tokens: torch.Tensor
    ) -> torch.Tensor:
        x = self.ups(x)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = nn.functional.interpolate(
                skip,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.film(x, txt_tokens)
        return x

class TextGuidedSegmentationModel(nn.Module):
    """Text-guided medical image segmentation model with dual-domain fusion."""
    def __init__(self, bert_type: str, vision_type: str):
        super().__init__()
        self.encoder = VisionEncoder(vision_type)
        self.text_encoder = FrozenTextEncoder(bert_type)
        self.spatial_dim = [7, 14, 28, 56]
        feature_dim = [768, 384, 192, 96]
        self.decoder16 = TextConditionedDecoder(
            in_channels=feature_dim[0],
            out_channels=feature_dim[1],
            spatial_size=self.spatial_dim[0],
            text_len=24
        )
        self.decoder8 = TextConditionedDecoder(
            in_channels=feature_dim[1],
            out_channels=feature_dim[2],
            spatial_size=self.spatial_dim[1],
            text_len=12
        )
        self.decoder4 = TextConditionedDecoder(
            in_channels=feature_dim[2],
            out_channels=feature_dim[3],
            spatial_size=self.spatial_dim[2],
            text_len=9
        )
        self.shallow112 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.shallow224 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        self.refine2 = HighResRefinementBlock(
            in_ch=feature_dim[3],
            skip_ch=48,
            out_ch=48,
            txt_dim=768
        )
        self.refine1 = HighResRefinementBlock(
            in_ch=48,
            skip_ch=24,
            out_ch=24,
            txt_dim=768
        )
        self.out = UnetOutBlock(
            spatial_dims=2,
            in_channels=24,
            out_channels=1
        )
    def forward(self, data):
        image, text = data
        if image.shape[1] == 1:
            image = repeat(image, "b 1 h w -> b c h w", c=3)
        skip112 = self.shallow112(image)
        skip224 = self.shallow224(image)
        img_out = self.encoder(image)
        img_feats = img_out["feature"]
        txt_out = self.text_encoder(text["input_ids"], text["attention_mask"])
        txt_feats = txt_out["feature"]
        txt_last = txt_feats[-1]
        if len(img_feats[0].shape) == 4:
            img_feats = img_feats[1:]
            img_feats = [rearrange(f, "b c h w -> b (h w) c") for f in img_feats]
        os32 = img_feats[3]
        os16 = self.decoder16(
            vis=os32,
            skip_vis=img_feats[2],
            txt=txt_last
        )
        os8 = self.decoder8(
            vis=os16,
            skip_vis=img_feats[1],
            txt=txt_last
        )
        os4 = self.decoder4(
            vis=os8,
            skip_vis=img_feats[0],
            txt=txt_last
        )
        os4_map = rearrange(
            os4,
            "b (h w) c -> b c h w",
            h=self.spatial_dim[-1],
            w=self.spatial_dim[-1]
        )
        x112 = self.refine2(
            x=os4_map,
            skip=skip112,
            txt_tokens=txt_last
        )
        x224 = self.refine1(
            x=x112,
            skip=skip224,
            txt_tokens=txt_last
        )
        out = self.out(x224)
        return out