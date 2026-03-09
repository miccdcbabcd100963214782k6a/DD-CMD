import argparse
import os
import random
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import utils.config as config
from net.creratemodel import CreateModel
from utils.dataset import SegData
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

class Tee:
    """Write console output to both terminal and file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

class EpochTimerAndBest(pl.Callback):
    """Prints per-epoch training time and tracks best validation metric."""
    def __init__(self, monitor: str):
        super().__init__()
        self.monitor = str(monitor)
        self._epoch_start_time = None
        self.best_val = float("-inf")
        self.best_epoch = -1
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._epoch_start_time = time.perf_counter()
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or self._epoch_start_time is None:
            return
        dur = time.perf_counter() - self._epoch_start_time
        if trainer.is_global_zero:
            print(f"[Epoch {trainer.current_epoch}] train time: {dur/60:.2f} min ({dur:.2f} s)")
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if self.monitor not in metrics or metrics[self.monitor] is None:
            return
        try:
            v = float(metrics[self.monitor].detach().cpu())
        except Exception:
            v = float(metrics[self.monitor])
        if v > self.best_val:
            self.best_val = v
            self.best_epoch = trainer.current_epoch
            if trainer.is_global_zero:
                print(f"New best {self.monitor}: {self.best_val:.4f} at epoch {self.best_epoch}")
    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero and self.best_epoch != -1:
            print(f"Best {self.monitor}: {self.best_val:.4f} @ epoch {self.best_epoch}")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _infer_dataset_name(args) -> str:
    """Try to infer dataset name from paths."""
    cand = []
    for k in ["train_csv_path", "train_root_path", "val_csv_path", "val_root_path", "test_csv_path", "test_root_path"]:
        if hasattr(args, k):
            v = getattr(args, k)
            if isinstance(v, str) and v:
                cand.append(v)
    for p in cand:
        parts = Path(p).as_posix().split("/")
        if "data" in parts:
            i = parts.index("data")
            if i + 1 < len(parts):
                return parts[i + 1]
        if parts[0] in (".", "") and len(parts) > 2 and parts[1] == "data":
            return parts[2]
    if hasattr(args, "model_save_filename") and args.model_save_filename:
        return str(args.model_save_filename).split("_")[0]
    return "dataset"

def _safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path

def _make_run_dir(args, config_path: str) -> Path:
    """Create timestamped run directory with structured subfolders."""
    base = Path(args.model_save_path)
    base.mkdir(parents=True, exist_ok=True)
    dataset = _infer_dataset_name(args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{dataset}_{args.model_save_filename}"
    run_dir = base / name
    if run_dir.exists():
        for i in range(1, 999):
            alt = base / f"{name}_{i:03d}"
            if not alt.exists():
                run_dir = alt
                break
    _safe_mkdir(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "snapshot").mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(config_path, run_dir / "snapshot" / Path(config_path).name)
    except Exception as e:
        print(f"[WARN] Could not copy config file: {e}")
    try:
        from net import model as net_model
        from net import decoder as net_decoder
        from net import creratemodel as net_createmodel
        shutil.copy2(Path(__file__).resolve(), run_dir / "snapshot" / "train.py")
        shutil.copy2(Path(net_model.__file__).resolve(), run_dir / "snapshot" / "model.py")
        shutil.copy2(Path(net_decoder.__file__).resolve(), run_dir / "snapshot" / "decoder.py")
        shutil.copy2(Path(net_createmodel.__file__).resolve(), run_dir / "snapshot" / "creratemodel.py")
    except Exception as e:
        print(f"[WARN] Could not snapshot code files: {e}")
    return run_dir

def _print_params_and_gflops(seg_model: torch.nn.Module, image_size=(224, 224), text_len=24):
    """Print model parameters and GFLOPs."""
    total = sum(p.numel() for p in seg_model.parameters())
    trainable = sum(p.numel() for p in seg_model.parameters() if p.requires_grad)
    print(f"Params: total={total/1e6:.3f} M | trainable={trainable/1e6:.3f} M")
    try:
        from thop import profile
        class _Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, image, input_ids, attention_mask):
                text = {"input_ids": input_ids, "attention_mask": attention_mask}
                return self.m([image, text])
        w = _Wrap(seg_model).cpu().eval()
        h, wimg = int(image_size[0]), int(image_size[1])
        img = torch.randn(1, 3, h, wimg)
        ids = torch.randint(low=0, high=1000, size=(1, int(text_len)))
        mask = torch.ones_like(ids)
        flops, params = profile(w, inputs=(img, ids, mask), verbose=False)
        print(f"GFLOPs (approx): {flops/1e9:.3f} | THOP params: {params/1e6:.3f} M")
    except Exception as e:
        print(f"[INFO] GFLOPs not computed (install thop or unsupported ops): {e}")

def get_cfg_and_path():
    parser = argparse.ArgumentParser(description="Train (image+text, DCT spectral calibration)")
    parser.add_argument("--config", default="./config/train.yaml", type=str, help="yaml config file")
    cli = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(cli.config)
    return cfg, cli.config

if __name__ == "__main__":
    args, config_path = get_cfg_and_path()
    print("cuda is used:", torch.cuda.is_available())
    set_seed(42)
    run_dir = _make_run_dir(args, config_path)
    args.model_save_path = str(run_dir)
    log_file_path = os.path.join(args.model_save_path, "training_console.log")
    log_f = open(log_file_path, "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)
    print(f"Console log will be saved to: {log_file_path}")
    print(f"Run dir: {args.model_save_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True)
    ds_train = SegData(
        csv_path=args.train_csv_path,
        root_path=args.train_root_path,
        tokenizer=tokenizer,
        mode="train",
        image_size=args.image_size,
    )
    ds_valid = SegData(
        csv_path=args.val_csv_path,
        root_path=args.val_root_path,
        tokenizer=tokenizer,
        mode="valid",
        image_size=args.image_size,
    )
    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model = CreateModel(args)
    try:
        _print_params_and_gflops(model.model, image_size=args.image_size, text_len=getattr(ds_train, "input_text_len", 24))
    except Exception as e:
        print(f"[WARN] Could not print Params/GFLOPs: {e}")
    monitor = getattr(args, "ckpt_monitor", "val_dice")
    mode = getattr(args, "ckpt_mode", "max")
    save_top_k = int(getattr(args, "save_top_k", 5))
    save_last = bool(getattr(args, "save_last", True))
    if save_top_k > 1:
        filename = f"{args.model_save_filename}-" + "{epoch:03d}-{" + f"{monitor}" + ":.4f}"
    else:
        filename = args.model_save_filename
    model_ckpt = ModelCheckpoint(
        dirpath=os.path.join(args.model_save_path, "checkpoints"),
        filename=filename,
        monitor=monitor,
        save_top_k=save_top_k,
        save_last=save_last,
        mode=mode,
        verbose=True,
    )
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=args.patience,
        mode=mode,
    )
    progress_bar = TQDMProgressBar(refresh_rate=10)
    epoch_timer = EpochTimerAndBest(monitor=monitor)
    tb_logger = TensorBoardLogger(save_dir=args.model_save_path, name="tb_logs")
    csv_logger = CSVLogger(save_dir=args.model_save_path, name="csv_logs")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = args.device if accelerator == "gpu" else 1
    trainer = pl.Trainer(
        logger=[tb_logger, csv_logger],
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[model_ckpt, early_stopping, epoch_timer, progress_bar],
        enable_progress_bar=True,
        log_every_n_steps=10,
        enable_model_summary=False,
    )
    print("==== start ====")
    trainer.fit(model, dl_train, dl_valid)
    print("==== finish ====")
    try:
        log_f.close()
    except Exception:
        pass