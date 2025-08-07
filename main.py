from __future__ import annotations

# ── std‑lib ─────────────────────────────────────────────────────────────
import argparse, pathlib, random, time, copy, os, itertools
from dataclasses import dataclass
from typing import List, Tuple

# ── third‑party ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import (
    DataLoader, Dataset, Subset, WeightedRandomSampler
)
from torchvision import transforms, models
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
from colorama import init as colorama_init
from torch.utils.checkpoint import checkpoint_sequential

# ────────────────────────────────────────────────────────────────────────
# 0.  Environment tuning
# ────────────────────────────────────────────────────────────────────────
colorama_init(autoreset=True)

# ────────────────────────────────────────────────────────────────────────
# 1.  Helpers
# ────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ────────────────────────────────────────────────────────────────────────
# 2.  Dataset
# ────────────────────────────────────────────────────────────────────────
class HAM10000Dataset(Dataset):
    def __init__(
        self,
        metadata_csv: pathlib.Path,
        images_dir: pathlib.Path,
        transform: transforms.Compose | None = None,
    ) -> None:
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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> Tuple[Metrics, List[int], List[int]]:
    model.eval()
    loss_sum = correct = total = 0
    preds, labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        loss = criterion(out, y)

        loss_sum += loss.item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        preds.extend(pred.cpu().tolist())
        labels.extend(y.cpu().tolist())

    return Metrics(loss_sum / total, correct / total), labels, preds


def train_epoch(
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    accum: int,
) -> Metrics:
    model.train()
    loss_sum = correct = total = 0

    for step, (x, y) in enumerate(
        tqdm(loader, desc=f"Epoch {epoch:02d}", colour="green")
    ):
        x, y = x.to(device), y.to(device)

        if step % accum == 0:
            optimizer.zero_grad(set_to_none=True)

        out  = model(x)
        loss = criterion(out, y) / accum
        loss.backward()

        if (step + 1) % accum == 0:
            optimizer.step()

        loss_sum += loss.item() * y.size(0) * accum
        correct  += (out.argmax(1) == y).sum().item()
        total    += y.size(0)

    return Metrics(loss_sum / total, correct / total)

# ────────────────────────────────────────────────────────────────────────
# 4.  Data loaders
# ────────────────────────────────────────────────────────────────────────
def build_loaders(
    args,
    ds_train: HAM10000Dataset,
    ds_eval: HAM10000Dataset,
    fold: int = 0,
):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y = ds_train.df["dx"]
    groups = ds_train.df["lesion_id"].fillna(ds_train.df["image_id"])
    train_idx, test_idx = list(
        sgkf.split(np.zeros(len(ds_train)), y, groups)
    )[fold]

    train_df = ds_train.df.iloc[train_idx]
    test_df  = ds_train.df.iloc[test_idx]
    val_mask = train_df.groupby("dx").sample(frac=0.2, random_state=42).index
    val_df   = train_df.loc[val_mask]
    train_df = train_df.drop(val_mask)

    train_ds = Subset(ds_train, train_df.index.to_numpy())
    val_ds   = Subset(ds_eval,  val_df.index.to_numpy())
    test_ds  = Subset(ds_eval,  test_df.index.to_numpy())

    class_counts = (
        train_df["dx"].value_counts().reindex(ds_train.classes, fill_value=0)
    ).to_numpy()
    class_counts = np.where(class_counts == 0, 1, class_counts)  # avoid /0
    weights = 1.0 / class_counts
    sample_weights = [weights[ds_train.class_to_idx[lbl]] for lbl in train_df["dx"]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    mk_loader = lambda d, shuffle=False, sampler=None: DataLoader(
        d,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=(args.device_type == "cuda"),
        prefetch_factor=4,
        persistent_workers=(args.workers > 0),
    )

    return (
        mk_loader(train_ds, sampler=sampler),
        mk_loader(val_ds),
        mk_loader(test_ds),
    )

# ────────────────────────────────────────────────────────────────────────
# 5.  Main
# ────────────────────────────────────────────────────────────────────────
def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser("Skin‑Cancer Training Script (FP32)")

    # —— paths ——————————————————————————————————————————
    parser.add_argument("--metadata", type=pathlib.Path,
                        default=pathlib.Path("~/Desktop/Data/HAM10000_metadata.csv").expanduser())
    parser.add_argument("--images-dir", type=pathlib.Path,
                        default=pathlib.Path("~/Desktop/Data/all_images").expanduser())
    parser.add_argument("--output",  type=pathlib.Path, default="mira.pth")
    parser.add_argument("--resume",  type=pathlib.Path,
                        help="checkpoint to resume training from")

    # —— core hyper‑params ————————————————————————————
    parser.add_argument("--model",
        choices=["resnet18", "resnet34",
                 "convnext_tiny", "convnext_large", "convnext_xl"],
        default="convnext_xl")
    parser.add_argument("--img-size",   type=int, default=768,
                        help="square crop fed to the network")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--lr",         type=float, default=3e-5)
    parser.add_argument("--accum",      type=int,   default=1)

    # —— misc ————————————————————————————————————————
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count()))
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args(argv)
    assert args.img_size > 0, "--img-size must be positive"

    # device ------------------------------------------------------------
    args.device_type = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    device = torch.device(args.device_type)
    print("Device:", device)

    # expand paths
    args.metadata   = args.metadata.expanduser()
    args.images_dir = args.images_dir.expanduser()
    if args.resume:
        args.resume = args.resume.expanduser()

    set_seed(args.seed)
    start_wall = time.time()

    # ── transforms ────────────────────────────────────────────────────
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # datasets & loaders ------------------------------------------------
    ds_train = HAM10000Dataset(args.metadata, args.images_dir, transform=train_tf)
    ds_eval  = HAM10000Dataset(args.metadata, args.images_dir, transform=eval_tf)
    train_loader, val_loader, test_loader = build_loaders(args, ds_train, ds_eval)

    # model -------------------------------------------------------------
    if args.model == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, len(ds_train.classes))
    elif args.model == "resnet34":
        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, len(ds_train.classes))
    elif args.model == "convnext_large":
        backbone = models.convnext_large(
            weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        )
        backbone.classifier[2] = nn.Linear(
            backbone.classifier[2].in_features, len(ds_train.classes)
        )
        blocks = list(backbone.features)

        def cp_forward_lg(x):
            return checkpoint_sequential(blocks, 4, x)

        backbone.features.forward = cp_forward_lg
    elif args.model == "convnext_xl":
        import timm
        backbone = timm.create_model(
            "convnext_xlarge", pretrained=True, num_classes=len(ds_train.classes)
        )
        backbone.set_grad_checkpointing(True)
    else:  # convnext_tiny
        backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        backbone.classifier[2] = nn.Linear(
            backbone.classifier[2].in_features, len(ds_train.classes)
        )

    model = backbone.to(device)
    if hasattr(torch, "compile") and device.type != "mps": # skip on Metal
        model = torch.compile(model, backend="eager")
        

    # optimisation ------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1_000_000)

    start_epoch = 0
    best_acc    = 0.0
    best_state  = copy.deepcopy(model.state_dict())  # fallback

    if args.resume and args.resume.is_file():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc    = ckpt.get("best_acc", 0.0)
        best_state  = ckpt["model"]
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    # training loop -----------------------------------------------------
    try:
        for epoch in itertools.count(start=start_epoch):
            train_metrics = train_epoch(
                epoch, model, train_loader, criterion, optimizer, device, args.accum
            )
            val_metrics, labels, preds = evaluate(model, val_loader, criterion, device)


            malignant_idx = torch.tensor(
                [ds_train.class_to_idx[c] for c in ["mel", "bcc", "akiec"]],
                device=device,
            )

            y_t   = torch.tensor(labels, device=device)
            p_t   = torch.tensor(preds,  device=device)

            is_malignant_y = torch.isin(y_t, malignant_idx)
            is_malignant_p = torch.isin(p_t, malignant_idx)

            TP = ( is_malignant_p &  is_malignant_y).sum().item()
            TN = (~is_malignant_p & ~is_malignant_y).sum().item()
            FP = ( is_malignant_p & ~is_malignant_y).sum().item()
            FN = (~is_malignant_p &  is_malignant_y).sum().item()

            sensitivity   = TP / (TP + FN + 1e-9)   # recall for cancer
            specificity   = TN / (TN + FP + 1e-9)   # true‑negative rate
            balanced_acc  = 0.5 * (sensitivity + specificity)
            scheduler.step()

            if val_metrics.acc > best_acc:
                best_acc   = val_metrics.acc
                best_state = copy.deepcopy(model.state_dict())

            print(
                f"Epoch {epoch:02d}  "
                f"train_loss={train_metrics.loss:.4f}  train_acc={train_metrics.acc:.4f}  "
                f"val_loss={val_metrics.loss:.4f}    val_acc={val_metrics.acc:.4f}  "
                f"sens={sensitivity:.3f}  spec={specificity:.3f}  bal_acc={balanced_acc:.3f}"
            )

            if epoch % 5 == 0:
                torch.save(
                    {
                        "epoch":     epoch,
                        "model":     model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_acc":  best_acc,
                    },
                    f"ckpt_epoch_{epoch}.pt",
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # checkpoint & test -------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output)
    print(f"Saved best model to {args.output}")
    model.load_state_dict(best_state)

    test_metrics, _, _ = evaluate(model, test_loader, criterion, device)
    print(
        f"\nTest loss={test_metrics.loss:.4f}  "
        f"test_acc={test_metrics.acc:.4f}  "
        f"(elapsed {(time.time()-start_wall)/60:.1f} min)"
    )

if __name__ == "__main__":
    main()

