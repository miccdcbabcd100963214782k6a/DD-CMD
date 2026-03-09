import os
from typing import Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandGaussianNoised,
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    RandRotated,
    Resized,
    ToTensord,
)


class SegData(Dataset):

    def __init__(
        self,
        csv_path: str,
        root_path: str,
        tokenizer,
        mode: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        input_text_len: int = 24,
    ):
        super().__init__()
        self.mode = mode
        self.root_path = root_path
        self.tokenizer = tokenizer
        self.image_size = list(image_size)
        self.input_text_len = int(input_text_len)

        self.data = pd.read_csv(csv_path)
        gt_dir = os.path.join(root_path, "GTs")
        if not os.path.isdir(gt_dir):
            raise FileNotFoundError(f"GTs folder not found: {gt_dir}")
        self.mask_list = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

        # ---- caption map ----
        # Prefer column names: Image + text/Description
        if "Image" in self.data.columns:
            key_col = "Image"
        else:
            key_col = self.data.columns[0]

        if "text" in self.data.columns:
            txt_col = "text"
        elif "Description" in self.data.columns:
            txt_col = "Description"
        else:
            # fallback: any non-key column
            candidates = [c for c in self.data.columns if c != key_col]
            if not candidates:
                raise ValueError(f"No caption column found in {csv_path}. Columns: {list(self.data.columns)}")
            txt_col = candidates[0]

        def _strip_mask_prefix(name: str) -> str:
            name = os.path.basename(str(name))
            return name.replace("mask_", "", 1) if name.startswith("mask_") else name

        self.caption_map: Dict[str, str] = {}
        for k, v in zip(self.data[key_col].tolist(), self.data[txt_col].tolist()):
            self.caption_map[_strip_mask_prefix(k)] = str(v)

        # ---- transforms ----
        keys = ["image", "gt"]
        if mode == "train":
            trans = Compose([
                LoadImaged(keys=keys, reader="PILReader"),
                EnsureChannelFirstd(keys=keys),
                RandZoomd(keys=keys, min_zoom=0.95, max_zoom=1.15, mode=["bicubic", "nearest"], prob=0.3),
                RandRotated(keys=keys, range_x=[-0.3, 0.3], keep_size=True, mode=["bicubic", "nearest"], prob=0.3),
                RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                RandRotate90d(keys=keys, prob=0.2, max_k=3),
                Resized(keys=["image"], spatial_size=self.image_size, mode="bicubic"),
                Resized(keys=["gt"], spatial_size=self.image_size, mode="nearest"),
                NormalizeIntensityd(keys=["image"], channel_wise=True),
                ToTensord(keys=["image", "gt"]),
            ])
        else:
            trans = Compose([
                LoadImaged(keys=keys, reader="PILReader"),
                EnsureChannelFirstd(keys=keys),
                Resized(keys=["image"], spatial_size=self.image_size, mode="bicubic"),
                Resized(keys=["gt"], spatial_size=self.image_size, mode="nearest"),
                NormalizeIntensityd(keys=["image"], channel_wise=True),
                ToTensord(keys=["image", "gt"]),
            ])

        self.transform = trans

    def __len__(self) -> int:
        return len(self.mask_list)

    def __getitem__(self, idx: int):
        mask_name = self.mask_list[idx]
        img_name = mask_name.replace("mask_", "", 1) if mask_name.startswith("mask_") else mask_name
        img_path = os.path.join(self.root_path, "img", img_name)
        gt_path = os.path.join(self.root_path, "GTs", mask_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Mask not found: {gt_path}")

        key = img_name
        if key not in self.caption_map and mask_name in self.caption_map:
            key = mask_name
        if key not in self.caption_map:
            raise KeyError(
                f"No caption found for '{img_name}' (or '{mask_name}'). "
                f"Example keys: {list(self.caption_map.keys())[:5]}"
            )
        caption = self.caption_map[key]

        token_output = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.input_text_len,
            truncation=True,
            return_tensors="pt",
        )
        text = {k: v.squeeze(0) for k, v in token_output.items()}

        data = {"image": img_path, "gt": gt_path}
        data = self.transform(data)
        image = data["image"].float()
        gt = data["gt"].float()

        if gt.ndim == 3 and gt.shape[0] > 1:
            gt = gt[0:1, ...]
        gt = torch.where(gt == 255, 1.0, gt)
        gt = (gt > 0.5).float()

        return [image, text], gt