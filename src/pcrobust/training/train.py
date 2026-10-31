"""
train.py — Train a point-cloud embedding backbone for the robustness study.

Supports PointNet, PointNet++, and DGCNN architectures. Trains a classifier
over MeshGalleryDataset's categories to shape an embedding space; the
embedding itself (not the classifier logits) is what downstream evaluation
scripts (confirmatory.py, noise_robustness.py, occlusion_robustness.py,
etc.) actually score, via cosine similarity between clean and perturbed
embeddings. Checkpoints load via pcrobust.models.loading.load_model().

Usage:
    python -m pcrobust.training.train --arch pointnet   --pretrained meder411_pointnet.pth --device cuda
    python -m pcrobust.training.train --arch pointnetpp --pretrained pointnetpp_yanx27.pth --device cuda
    python -m pcrobust.training.train --arch dgcnn      --pretrained dgcnn_modelnet40.pth  --device cuda
"""

import argparse
import json
import random
import sys
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from pcrobust.datasets import MeshGalleryDataset, checkpoint_dir, domain_root
from pcrobust.models import REGISTRY, get_model, load_backbone
from pcrobust.utils import _augment, _sample


# ── TNet regularisation (PointNet-specific; skipped when aux is None) ──────────

def tnet_reg(T: torch.Tensor) -> torch.Tensor:
    """Regularise feature transform towards orthogonality: ||T·Tᵀ - I||."""
    B, k, _ = T.shape
    I    = torch.eye(k, device=T.device).unsqueeze(0).expand(B, -1, -1)
    diff = torch.bmm(T, T.transpose(1, 2)) - I
    return diff.norm(dim=(1, 2)).mean()


# ── Internal validation split (unified-recipe selection, 2026-08-13) ──────────
# Mirrors the invariance-loss trainers' _stratified_split/_Subset exactly
# (pcrobust.training.invariance), so both training conditions can share one
# checkpoint-selection criterion. Only active when --val_frac > 0; the default
# (0) reproduces the published runs: no split, selection on training accuracy.


def _stratified_split(labels, val_frac: float, seed: int):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i, lab in enumerate(labels):
        by_class[lab].append(i)
    train_idx, val_idx = [], []
    for lab, idxs in by_class.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = max(1, round(len(idxs) * val_frac))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx


class _GallerySubset(Dataset):
    """Index-selected view of a loaded MeshGalleryDataset with its own
    augment/repeats behavior -- same as the invariance-loss trainers'
    _Subset. `aug_mode="invreg_views"` (W4 control arm, 2026-08-13)
    additionally perturbs each augmented view with the invariance loss's
    own generators at the loss's per-step view composition -- per batch of
    32 the invreg trainers process 32 clean + 32 rotated + 32 noisy + 4
    occluded views, so each view here independently draws
    rotation/noise/occlusion/none at probabilities 32:32:32:4 (/100).
    Combined with a tripled --repeats this matches the invreg regime's
    perturbed-view budget while training with classification loss only:
    the arm that separates "the loss helps" from "seeing perturbed views
    helps"."""

    def __init__(self, base: MeshGalleryDataset, indices, augment: bool, repeats: int,
                 aug_mode: str = "default"):
        self.base = base
        self.indices = list(indices)
        self.do_augment = augment
        self.repeats = repeats if augment else 1
        self.n_points = base.n_points
        self.aug_mode = aug_mode
        if aug_mode == "invreg_views":
            # The invariance loss's own 1-arg generators (random
            # level/direction drawn inside), exactly what the invreg
            # trainer feeds its loss.
            from pcrobust.corruptions.invreg import (random_rotation,
                                                     add_noise, occlude)
            self._perturb = (random_rotation, add_noise, occlude)

    def __getitem__(self, idx):
        i = self.indices[idx % len(self.indices)]
        pts = self.base.raw[i].copy()
        if self.do_augment:
            pts = _augment(pts)
        if self.aug_mode == "invreg_views":
            random_rotation, add_noise, occlude = self._perturb
            r = random.random() * 100
            if r < 32:
                pts = random_rotation(pts)
            elif r < 64:
                pts = add_noise(pts)
            elif r < 68:
                pts = occlude(pts)
        pts = _sample(pts, self.n_points)
        return torch.from_numpy(pts), self.base.labels[i]

    def __len__(self):
        return len(self.indices) * self.repeats


# ── Training loop ──────────────────────────────────────────────────────────────

_HEADS_PREFIX = ("embed_head", "cls_head")


def train(args):
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    print(f"[train] arch: {args.arch}  device: {device}")

    if args.aug_mode != "default" and args.val_frac <= 0:
        sys.exit("[train] --aug_mode requires --val_frac > 0 (the unified-recipe path)")
    if args.val_frac > 0:
        dataset = MeshGalleryDataset(args.gallery, n_points=1024, augment=True, repeats=1)
        train_idx, val_idx = _stratified_split(dataset.labels, args.val_frac, args.split_seed)
        train_ds = _GallerySubset(dataset, train_idx, augment=True, repeats=args.repeats,
                                  aug_mode=args.aug_mode)
        val_ds   = _GallerySubset(dataset, val_idx, augment=False, repeats=1)
        loader = DataLoader(train_ds, batch_size=args.batch_size,
                            shuffle=True, num_workers=0, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
        print(f"[train] internal split: {len(train_idx)} train / {len(val_idx)} val "
              f"(stratified per class, val_frac={args.val_frac}, seed={args.split_seed}) -- "
              f"NOT the same as data/test/, kept out of checkpoint selection to avoid leakage")
    else:
        dataset = MeshGalleryDataset(args.gallery, n_points=1024, augment=True, repeats=args.repeats)
        loader  = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0, drop_last=False)
        val_loader = None

    model = get_model(args.arch, num_classes=len(dataset.classes), embed_dim=128).to(device)

    if args.pretrained:
        load_backbone(args.arch, model, args.pretrained)

    # Phase 1 — freeze backbone, warm up heads only
    frozen = bool(args.pretrained and args.freeze_epochs > 0)
    if frozen:
        for name, p in model.named_parameters():
            if not any(name.startswith(h) for h in _HEADS_PREFIX):
                p.requires_grad_(False)
        n_frozen = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
        print(f"[train] backbone frozen for first {args.freeze_epochs} epochs "
              f"({n_frozen} tensors)")

    opt   = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, best_state = 0.0, None

    for epoch in range(1, args.epochs + 1):

        # Phase 2 — unfreeze backbone and switch to fine-tune LR
        if frozen and epoch == args.freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad_(True)
            opt   = torch.optim.Adam(model.parameters(),
                                     lr=args.lr * 0.1, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs - args.freeze_epochs)
            frozen = False
            print(f"[train] backbone unfrozen -- fine-tuning all layers "
                  f"(lr={args.lr * 0.1:.1e})")

        model.train()
        total_loss, correct, n = 0.0, 0, 0

        for pts, labels in loader:
            pts    = pts.to(device).transpose(1, 2)   # (B, 3, N)
            labels = labels.to(device)

            opt.zero_grad()
            logits, _, aux = model(pts)
            loss = F.cross_entropy(logits, labels)
            if aux is not None:
                loss = loss + 0.001 * tnet_reg(aux)
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            n          += len(labels)

        sched.step()
        acc = correct / n

        if val_loader is not None:
            model.eval()
            v_correct, v_n = 0, 0
            with torch.no_grad():
                for pts, labels in val_loader:
                    pts    = pts.to(device).transpose(1, 2)
                    labels = labels.to(device)
                    logits, _, _ = model(pts)
                    v_correct += (logits.argmax(1) == labels).sum().item()
                    v_n       += len(labels)
            sel_acc = v_correct / v_n
        else:
            sel_acc = acc

        if sel_acc > best_acc:
            best_acc   = sel_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 25 == 0 or epoch == args.epochs:
            val_str = f"  val_acc={sel_acc:.3f}" if val_loader is not None else ""
            print(f"  epoch {epoch:4d}/{args.epochs}  "
                  f"loss={total_loss/n:.4f}  acc={acc:.3f}{val_str}  (best={best_acc:.3f})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{args.arch}.pth"
    torch.save({
        "arch":        args.arch,
        "state_dict":  best_state,
        "classes":     dataset.classes,
        "embed_dim":   128,
        "num_classes": len(dataset.classes),
    }, ckpt)
    (out / "classes.json").write_text(
        json.dumps(dataset.classes, indent=2), encoding="utf-8"
    )
    print(f"\n[train] Best acc : {best_acc:.3f}")
    print(f"[train] Checkpoint: {ckpt}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Train a point-cloud embedding backbone")
    p.add_argument("--arch",        default="pointnet",
                   choices=list(REGISTRY),
                   help="Backbone architecture (default: pointnet)")
    p.add_argument("--domain",      default="modelnet40",
                   help="Domain whose gallery and checkpoint directory are the defaults")
    p.add_argument("--gallery",     default=None,
                   help="Directory of *_gallery.ply files (default: the --domain gallery)")
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--repeats",     type=int,   default=100,
                   help="Augmented views per scan per epoch cycle (default: 100)")
    p.add_argument("--out",         default=None,
                   help="Directory to save checkpoint, saved as <arch>.pth "
                        "(default: the zoo cache location for --domain)")
    p.add_argument("--device",      default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--pretrained",  default=None,
                   help="Path to a pretrained backbone checkpoint")
    p.add_argument("--freeze_epochs", type=int, default=10,
                   help="Freeze backbone for N epochs when using --pretrained, "
                        "then fine-tune all layers at lr x0.1 (0 = fine-tune immediately)")
    p.add_argument("--val_frac",   type=float, default=0.0,
                   help="If > 0, carve a stratified validation split of this fraction "
                        "out of the gallery and select the checkpoint by clean "
                        "validation accuracy (matches the invariance-loss trainers). "
                        "Default 0 = the published behavior: no split, selection on "
                        "training accuracy.")
    p.add_argument("--split_seed", type=int, default=42,
                   help="Seed for the --val_frac stratified split (same default as "
                        "the invariance-loss trainers).")
    p.add_argument("--aug_mode", default="default", choices=["default", "invreg_views"],
                   help="'invreg_views' (W4 control arm): each augmented view is "
                        "additionally perturbed by the invariance loss's own "
                        "generators at the loss's 32:32:32:4 "
                        "clean/rotation/noise/occlusion view composition; train "
                        "with classification loss only. Pair with a tripled "
                        "--repeats to match the invreg regime's view budget.")
    args = p.parse_args(argv)
    if args.gallery is None:
        args.gallery = str(domain_root(args.domain) / "data" / "gallery")
    if args.out is None:
        args.out = str(checkpoint_dir(args.domain))
    train(args)


if __name__ == "__main__":
    main()
