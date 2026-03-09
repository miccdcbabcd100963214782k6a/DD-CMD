import argparse
import os
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import utils.config as config
from net.creratemodel import CreateModel
from utils.dataset import SegData

@torch.no_grad()
def save_pred_mask(prob: torch.Tensor, out_path: str, threshold: float = 0.5):
    """Save predicted probability mask as binary PNG."""
    prob = prob.float().cpu()
    mask = (prob > threshold).float().numpy()[0]
    mask = (mask * 255).astype(np.uint8)
    Image.fromarray(mask).save(out_path)

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate / test (image+text segmentation)")
    parser.add_argument("--config", default="./config/train.yaml", type=str, help="yaml config file")
    parser.add_argument("--ckpt", required=True, type=str, help="path to .ckpt from Lightning")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"], help="which split to evaluate")
    parser.add_argument("--save_preds", action="store_true", help="save predicted masks as PNGs")
    parser.add_argument("--pred_dir", default="./predictions", type=str, help="output folder for masks")
    parser.add_argument("--threshold", default=0.5, type=float, help="binarization threshold")
    return parser.parse_args()

def _torch_load_ckpt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def _strip_profile_keys(state_dict: dict) -> dict:
    bad = ("total_ops", "total_params")
    cleaned = {}
    removed = 0
    for k, v in state_dict.items():
        if any(b in k for b in bad):
            removed += 1
            continue
        cleaned[k] = v
    if removed > 0:
        print(f"[INFO] Stripped {removed} profiling keys (total_ops/total_params) from checkpoint.")
    return cleaned

def main():
    cli = get_args()
    args = config.load_cfg_from_cfg_file(cli.config)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True)
    if cli.split == "train":
        csv_path, root_path = args.train_csv_path, args.train_root_path
        mode = "train"
    elif cli.split == "valid":
        csv_path, root_path = args.val_csv_path, args.val_root_path
        mode = "valid"
    else:
        csv_path, root_path = args.test_csv_path, args.test_root_path
        mode = "valid"
    ds = SegData(
        csv_path=csv_path,
        root_path=root_path,
        tokenizer=tokenizer,
        mode=mode,
        image_size=args.image_size,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    model = CreateModel(args)
    ckpt = _torch_load_ckpt(cli.ckpt)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = _strip_profile_keys(state)
    try:
        model.load_state_dict(state, strict=True)
        print("[INFO] Loaded checkpoint with strict=True.")
    except RuntimeError as e:
        print("[WARN] strict=True failed. Retrying with strict=False.")
        print(f"[WARN] strict=True error:\n{e}\n")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[WARN] Loaded with strict=False. missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("[WARN] Missing keys (first 30):", missing[:30])
        if len(unexpected) > 0:
            print("[WARN] Unexpected keys (first 30):", unexpected[:30])
    model.eval()
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )
    results = trainer.test(model, dataloaders=dl, verbose=True)
    print("Test results:", results)
    if cli.save_preds:
        Path(cli.pred_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving predicted masks to: {cli.pred_dir}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        with torch.no_grad():
            for i, (x, _) in enumerate(dl):
                image, text = x
                image = image.to(device)
                text = {k: v.to(device) for k, v in text.items()}
                logits = model([image, text])
                prob = torch.sigmoid(logits).detach()
                out_path = os.path.join(cli.pred_dir, f"pred_{i:05d}.png")
                save_pred_mask(prob[0], out_path, threshold=cli.threshold)
        print("[INFO] Done saving predictions.")

if __name__ == "__main__":
    main()