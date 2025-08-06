from __future__ import annotations
import argparse, pathlib, random, time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import StratifiedGroupKFold

from tqdm import tqdm
from colorama import init as colorama_init             # coloured terminal text
import os
from torch.utils.checkpoint import checkpoint_sequential
from torchinfo import summary                          # layer‑by‑layer model summary

# ────────────────────────────────────────────────────────────────────────
# 0.  Environment tuning
# ────────────────────────────────────────────────────────────────────────
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())
colorama_init(autoreset=True)

# ────────────────────────────────────────────────────────────────────────
# 1.  Helpers
# ────────────────────────────────────────────────────────────────────────
def show_model_stats(model: torch.nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp32_mb = total * 4 / 1e6
    fp16_mb = total * 2 / 1e6
    print(f"Parameters  | total={total:,}  trainable={trainable:,}")
    print(f"Checkpoint  | fp32≈{fp32_mb:.1f} MB  fp16≈{fp16_mb:.1f} MB")

def show_model_summary(model: torch.nn.Module, img_size: int, batch: int = 1) -> None:
    summary(model,
            input_size=(batch, 3, img_size, img_size),
            col_names=("input_size", "output_size", "num_params"),
            row_settings=("var_names",),
            depth=3)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ────────────────────────────────────────────────────────────────────────
# 2.  Dataset
# ────────────────────────────────────────────────────────────────────────
class HAM10000Dataset(Dataset):
    def __init__(self,
                 metadata_csv: pathlib.Path,
                 images_dir: pathlib.Path,
                 transform: transforms.Compose | None = None) -> None:
        self.df = pd.read_csv(metadata_csv)
        self.transform = transform
        self.images_dir = images_dir
        self.classes = sorted(self.df["dx"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:                 # pragma: no cover
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img = Image.open(self.images_dir / f"{row.image_id}.jpg").convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[row.dx]
        return img, label

# ────────────────────────────────────────────────────────────────────────
# 3.  Training / evaluation
# ────────────────────────────────────────────────────────────────────────
@dataclass
class Metrics:
    loss: float
    acc:  float

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion, device) -> Tuple[Metrics, List[int], List[int]]:
    model.eval()
    loss_sum = correct = total = 0
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        preds.extend(pred.cpu().tolist())
        labels.extend(y.cpu().tolist())
    return Metrics(loss_sum / total, correct / total), labels, preds

def train_epoch(epoch: int, model, loader, criterion,
                optimizer, scaler, device, accum: int) -> Metrics:
    model.train()
    loss_sum = correct = total = 0
    for step, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch:02d}", colour="green")):
        x, y = x.to(device), y.to(device)
        if step % accum == 0:
            optimizer.zero_grad(set_to_none=True)

        use_amp = device.type == "cuda"          # disable autocast on MPS
        if device.type == "mps":
            x = x.half()                         # inputs must match weights
        with torch.amp.autocast(device_type=device.type,
                                dtype=torch.float16,
                                enabled=use_amp):
            out = model(x)
            loss = criterion(out, y) / accum

        if scaler:
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if (step + 1) % accum == 0:
                optimizer.step()

        loss_sum += loss.item() * y.size(0) * accum
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return Metrics(loss_sum / total, correct / total)

# ────────────────────────────────────────────────────────────────────────
# 4.  Data loaders
# ────────────────────────────────────────────────────────────────────────
def build_loaders(args, ds_train: HAM10000Dataset,
                  ds_eval: HAM10000Dataset, fold: int = 0):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y = ds_train.df["dx"]
    groups = ds_train.df["lesion_id"].fillna(ds_train.df["image_id"])
    train_idx, test_idx = list(
        sgkf.split(np.zeros(len(ds_train)), y, groups))[fold]
    train_df = ds_train.df.iloc[train_idx].reset_index(drop=True)
    test_df = ds_train.df.iloc[test_idx].reset_index(drop=True)

    val_mask = train_df.groupby("dx").sample(frac=0.2, random_state=42).index
    val_df = train_df.loc[val_mask].reset_index(drop=True)
    train_df = train_df.drop(val_mask).reset_index(drop=True)

    train_ds = Subset(ds_train, train_df.index.to_numpy())
    val_ds = Subset(ds_eval, val_df.index.to_numpy())
    test_ds = Subset(ds_eval, test_df.index.to_numpy())

    class_counts = train_df["dx"].value_counts().reindex(ds_train.classes,
                                                          fill_value=0).to_numpy()
    weights = 1.0 / class_counts
    sample_weights = [weights[ds_train.class_to_idx[lbl]]
                      for lbl in train_df["dx"]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                    replacement=True)

    mk_loader = lambda d, shuffle=False, sampler=None: DataLoader(
        d, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=shuffle, sampler=sampler, pin_memory=False)

    return (mk_loader(train_ds, sampler=sampler),
            mk_loader(val_ds),
            mk_loader(test_ds))

# ────────────────────────────────────────────────────────────────────────
# 5.  Main
# ────────────────────────────────────────────────────────────────────────
def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser("Skin‑Cancer Training Script (v2)")
    parser.add_argument("--metadata", type=pathlib.Path, required=True)
    parser.add_argument("--images-dir", type=pathlib.Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model",
                        choices=["resnet18", "resnet34",
                                 "convnext_tiny", "convnext_large",
                                 "convnext_xl"],
                        default="resnet34")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--output", type=pathlib.Path, default="best_model.pth")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    start = time.time()
    set_seed(args.seed)

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    # transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    ds_train = HAM10000Dataset(args.metadata, args.images_dir, transform=train_tf)
    ds_eval = HAM10000Dataset(args.metadata, args.images_dir, transform=eval_tf)
    train_loader, val_loader, test_loader = build_loaders(
        args, ds_train, ds_eval)

    # model
    if args.model == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, len(ds_train.classes))
    elif args.model == "resnet34":
        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, len(ds_train.classes))
    elif args.model == "convnext_large":
        backbone = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
        backbone.classifier[2] = nn.Linear(backbone.classifier[2].in_features,
                                           len(ds_train.classes))
        # 4‑way checkpoint
        blocks = list(backbone.features)
        def cp_forward_lg(x): return checkpoint_sequential(blocks, 4, x)
        backbone.features.forward = cp_forward_lg
    elif args.model == "convnext_xl":
        import timm  # timm provides ConvNeXt‑XL

        # Build ConvNeXt‑XL with ImageNet‑22k pre‑training and a new 7‑class head
        backbone = timm.create_model(
            "convnext_xlarge",
            pretrained=True,
            num_classes=len(ds_train.classes)
        )

        # Enable timm’s built‑in 8‑way gradient checkpointing
        backbone.set_grad_checkpointing(True)
    else:  # convnext_tiny
        backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        backbone.classifier[2] = nn.Linear(backbone.classifier[2].in_features,
                                           len(ds_train.classes))

    model = backbone.to(device)
    if device.type == "mps":          # full‑FP16 path for Apple GPUs
        model = model.half()

    # diagnostics
    show_model_stats(model)
    show_model_summary(model, args.img_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(args.epochs):
        train_metrics = train_epoch(epoch, model, train_loader,
                                    criterion, optimizer, scaler,
                                    device, args.accum)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}  "
              f"train_loss={train_metrics.loss:.4f}  train_acc={train_metrics.acc:.4f}  "
              f"val_loss={val_metrics.loss:.4f}    val_acc={val_metrics.acc:.4f}")

    # ... rest of main (saving, test eval, etc.) ...

if __name__ == "__main__":
    main()