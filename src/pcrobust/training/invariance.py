"""Invariance-regularized (invreg) training: the paper's intervention recipe.

Trains any benchmark architecture with

    loss = cls_loss + lambda_inv * (inv_rotation + inv_noise + inv_occlusion)

where each inv_* term is ``1 - cos(emb(clean), emb(perturbed))`` on a
rotated, noised, or occluded copy of the same batch (occlusion on the first
``--occ_subbatch`` samples only, it is the expensive one). The architecture
class is unchanged; only the training objective differs from the baseline
recipe in :mod:`pcrobust.training.train`.

Recipes, selected by flags, each reproducing the paper's checkpoints:

  ``<arch>_invreg``             default: lambda_inv=0.5 from epoch 1
  ``<arch>_invreg_warmup``      --lambda_inv_warmup_epochs 10: lambda_inv
                                held at 0 then ramped linearly, which fixes a
                                representation-collapse stall seen in
                                RISurConv (val acc pinned at chance, inv_loss
                                collapsing toward zero)
  ``<arch>_invreg_pose270``     --augment pose270 / pose270yaw: wider rotation
  ``<arch>_invreg_pose270yaw``  augmentation in the training views

Checkpoint selection uses a stratified validation split carved out of the
training gallery (``--val_frac``), never the held-out test split.

Usage:
    python -m pcrobust.training.invariance --arch pointnet --device cuda
    python -m pcrobust.training.invariance --arch risurconv --lambda_inv_warmup_epochs 10 --device cuda
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pcrobust.corruptions.invreg import add_noise, occlude, random_rotation
from pcrobust.datasets import checkpoint_dir, domain_root
from pcrobust.models import REGISTRY, get_model

# Recipe parity with the published checkpoints. The per-architecture
# training runs differed in two small ways that this trainer reproduces
# rather than normalizes:
#
# - PointNet's T-Net orthogonality term (weight 0.001, clean view only, as
#   in the baseline recipe) is part of the plain invreg recipe. The warmup
#   and pose270 runs were trained without it.
# - Gradient accumulation scales each step's loss by B / batch_size. That is
#   1.0 for every full batch, but it also down-weights the last, partial
#   batch of each epoch. The plain invreg runs for the architectures below
#   predate accumulation and did not scale it; every other run did.
_TNET_WEIGHT = 0.001
_UNSCALED_LAST_BATCH = {"dgcnn", "riconvpp", "risurconv", "pointtransformer", "vn_pointnet"}

_AUGMENT_MODULES = {
    "default": "pcrobust.utils",
    "pose270": "pcrobust.training.augment_pose270",
    "pose270yaw": "pcrobust.training.augment_pose270yaw",
}


def _augmentation(name: str):
    import importlib

    mod = importlib.import_module(_AUGMENT_MODULES[name])
    return mod._augment, mod._sample


class _Subset(Dataset):
    """A subset of an already-loaded MeshGalleryDataset's raw points/labels,
    selected by index, with its own augment/repeats behavior. Carves an
    internal train/val split out of one loaded gallery without splitting
    files on disk."""

    def __init__(self, base, indices, augment: bool, repeats: int,
                 augment_fn, sample_fn):
        self.base = base
        self.indices = list(indices)
        self.do_augment = augment
        self.repeats = repeats if augment else 1
        self.n_points = base.n_points
        self.augment_fn = augment_fn
        self.sample_fn = sample_fn

    def __len__(self):
        return len(self.indices) * self.repeats

    def __getitem__(self, idx):
        i = self.indices[idx % len(self.indices)]
        pts = self.base.raw[i].copy()
        if self.do_augment:
            pts = self.augment_fn(pts)
        pts = self.sample_fn(pts, self.n_points)
        return torch.from_numpy(pts), self.base.labels[i]


def _tnet_reg(T: torch.Tensor) -> torch.Tensor:
    """Regularise PointNet's feature transform towards orthogonality:
    ||T.T^t - I||, as in the baseline recipe."""
    B, k, _ = T.shape
    I = torch.eye(k, device=T.device).unsqueeze(0).expand(B, -1, -1)
    diff = torch.bmm(T, T.transpose(1, 2)) - I
    return diff.norm(dim=(1, 2)).mean()


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


def _perturb_batch(pts_np: np.ndarray, kind: str, n_apply: int = None) -> np.ndarray:
    n_apply = pts_np.shape[0] if n_apply is None else n_apply
    out = pts_np.copy()
    for i in range(n_apply):
        if kind == "rotation":
            out[i] = random_rotation(pts_np[i])
        elif kind == "noise":
            out[i] = add_noise(pts_np[i])
        elif kind == "occlusion":
            out[i] = occlude(pts_np[i])
        else:
            raise ValueError(kind)
    return out


def _occlude_worker(pt: np.ndarray) -> np.ndarray:
    return occlude(pt)


def _lambda_inv_at(epoch: int, target: float, warmup_epochs: int) -> float:
    """Linear warmup: 0.0 at epoch 1, ramping to `target` by epoch
    `warmup_epochs + 1`, held at `target` thereafter. warmup_epochs=0
    disables the warmup (lambda_inv = target from epoch 1)."""
    if warmup_epochs <= 0:
        return target
    frac = min(1.0, (epoch - 1) / warmup_epochs)
    return target * frac


@torch.no_grad()
def _eval_clean_acc(model, loader, device):
    model.eval()
    correct, n = 0, 0
    for pts, labels in loader:
        labels = labels.to(device)
        x = pts.to(device).transpose(1, 2)
        logits, _, _ = model(x)
        correct += (logits.argmax(1) == labels).sum().item()
        n += labels.shape[0]
    model.train()
    return correct / n


def checkpoint_key(arch: str, warmup_epochs: int = 0, augment: str = "default") -> str:
    """The checkpoint key, and file stem inside the domain's checkpoint
    directory: ``<arch>_invreg[_warmup][_pose270|_pose270yaw]``."""
    key = f"{arch}_invreg"
    if warmup_epochs > 0:
        key += "_warmup"
    if augment != "default":
        key += f"_{augment}"
    return key


def arch_field(key: str, domain: str = "modelnet40") -> str:
    """The ``arch`` value stored inside the checkpoint. Published checkpoints
    for domains other than ModelNet40 carry a ``_<domain>`` suffix there."""
    return key if domain == "modelnet40" else f"{key}_{domain}"


def train(args):
    _t_start = time.time()
    plain = args.lambda_inv_warmup_epochs <= 0 and args.augment == "default"
    tnet_weight = (_TNET_WEIGHT if args.arch == "pointnet" and plain else 0.0) \
        if args.tnet_weight is None else args.tnet_weight
    scale_last_batch = not (plain and args.micro_batch_size is None
                            and args.arch in _UNSCALED_LAST_BATCH)
    if args.augment != "default" and args.micro_batch_size is None:
        scale_last_batch = False  # the pose270 runs predate accumulation
    augment_fn, sample_fn = _augmentation(args.augment)

    _KEY = checkpoint_key(args.arch, args.lambda_inv_warmup_epochs, args.augment)
    _ARCH = arch_field(_KEY, args.domain)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[invreg] key: {_ARCH}  domain: {args.domain}  device: {device}  "
          f"epochs={args.epochs} repeats={args.repeats} batch_size={args.batch_size} "
          f"lambda_inv={args.lambda_inv} (warmup {args.lambda_inv_warmup_epochs} epochs) "
          f"tnet_weight={tnet_weight}", flush=True)

    from pcrobust.datasets import MeshGalleryDataset
    gallery = args.gallery or str(domain_root(args.domain) / "data" / "gallery")
    full = MeshGalleryDataset(gallery, n_points=1024, augment=True, repeats=1)
    train_idx, val_idx = _stratified_split(full.labels, val_frac=args.val_frac, seed=args.split_seed)
    print(f"[invreg] internal split: {len(train_idx)} train / {len(val_idx)} val "
          f"(stratified per class, val_frac={args.val_frac}, seed={args.split_seed}) -- "
          f"NOT the same as data/test/, kept out of checkpoint selection to avoid leakage", flush=True)

    train_ds = _Subset(full, train_idx, augment=True, repeats=args.repeats,
                       augment_fn=augment_fn, sample_fn=sample_fn)
    val_ds = _Subset(full, val_idx, augment=False, repeats=1,
                     augment_fn=augment_fn, sample_fn=sample_fn)
    print(f"[invreg] {len(train_ds):,} train samples/epoch ({args.repeats}x aug), "
          f"{len(val_ds)} val samples (clean, no aug)", flush=True)

    # Gradient accumulation: --micro_batch_size samples per forward pass,
    # one optimizer step per --batch_size samples. Each step runs 4 forward
    # passes (clean/rot/noise/occ) before backward(), so peak memory scales
    # with the micro batch. Unset = no accumulation.
    micro_bs = args.micro_batch_size or args.batch_size
    if args.batch_size % micro_bs != 0:
        print(f"[invreg] WARNING: batch_size={args.batch_size} is not a multiple of "
              f"micro_batch_size={micro_bs} -- the final accumulation group each epoch "
              f"will be smaller than the rest.", flush=True)
    train_loader = DataLoader(train_ds, batch_size=micro_bs, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=micro_bs, shuffle=False,
                            num_workers=0, drop_last=False)

    occ_workers = min(args.occ_subbatch, os.cpu_count() or 1)
    occ_pool = ProcessPoolExecutor(max_workers=occ_workers)
    print(f"[invreg] occlusion generation parallelized across "
          f"{occ_workers} worker process(es) ({os.cpu_count()} cores available)", flush=True)

    model = get_model(args.arch, num_classes=len(full.classes), embed_dim=128).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[invreg] total parameters: {n_params:,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out = Path(args.out) if args.out else checkpoint_dir(args.domain)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{_KEY}.pth"
    (out / "classes.json").write_text(json.dumps(full.classes, indent=2), encoding="utf-8")

    best_val_acc, best_epoch = 0.0, 0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        lambda_inv_eff = _lambda_inv_at(epoch, args.lambda_inv, args.lambda_inv_warmup_epochs)
        total_loss, total_cls, total_inv, correct, n = 0.0, 0.0, 0.0, 0, 0
        _epoch_t0 = time.time()
        _n_batches = len(train_loader)
        accum_n = 0
        opt.zero_grad()

        for bi, (pts, labels) in enumerate(train_loader):
            B = pts.shape[0]
            labels = labels.to(device)
            clean = pts.to(device).transpose(1, 2)

            pts_np = pts.numpy()
            rot_np = _perturb_batch(pts_np, "rotation")
            noise_np = _perturb_batch(pts_np, "noise")
            occ_n = min(args.occ_subbatch, B)
            occ_np = np.stack(list(occ_pool.map(_occlude_worker, [pts_np[i] for i in range(occ_n)])))

            if bi % 10 == 0:
                print(f"    [heartbeat] epoch {epoch} batch {bi}/{_n_batches}  "
                      f"{time.time()-_epoch_t0:.1f}s elapsed this epoch  "
                      f"lambda_inv={lambda_inv_eff:.4f}", flush=True)

            rot = torch.from_numpy(rot_np).to(device).transpose(1, 2)
            noise = torch.from_numpy(noise_np).to(device).transpose(1, 2)
            occ = torch.from_numpy(occ_np).to(device).transpose(1, 2)

            logits, emb_clean, aux = model(clean)
            _, emb_rot, _ = model(rot)
            _, emb_noise, _ = model(noise)
            _, emb_occ, _ = model(occ)

            cls_loss = F.cross_entropy(logits, labels)
            if tnet_weight and aux is not None:
                cls_loss = cls_loss + tnet_weight * _tnet_reg(aux)

            inv_r = (1.0 - F.cosine_similarity(emb_clean, emb_rot, dim=1)).mean()
            inv_n = (1.0 - F.cosine_similarity(emb_clean, emb_noise, dim=1)).mean()
            inv_o = (1.0 - F.cosine_similarity(emb_clean[:occ_n], emb_occ, dim=1)).mean()
            inv_loss = inv_r + inv_n + inv_o

            loss = cls_loss + lambda_inv_eff * inv_loss

            # Scale by this micro-batch's share of the logical batch so the
            # accumulated gradient equals the average over the logical batch.
            (loss * (B / args.batch_size) if scale_last_batch else loss).backward()
            accum_n += B
            if accum_n >= args.batch_size or bi == _n_batches - 1:
                opt.step()
                opt.zero_grad()
                accum_n = 0

            total_loss += loss.item() * B
            total_cls += cls_loss.item() * B
            total_inv += inv_loss.item() * B
            correct += (logits.argmax(1) == labels).sum().item()
            n += B

        sched.step()
        train_acc = correct / n
        val_acc = _eval_clean_acc(model, val_loader, device)

        if val_acc > best_val_acc + args.min_delta:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            # Saved immediately so an interrupted run keeps its best checkpoint.
            torch.save({
                "arch": _ARCH,
                "state_dict": model.state_dict(),
                "classes": full.classes,
                "embed_dim": 128,
                "num_classes": len(full.classes),
            }, ckpt)
        else:
            epochs_no_improve += 1

        print(f"  epoch {epoch:4d}/{args.epochs}  lambda_inv={lambda_inv_eff:.4f}  "
              f"loss={total_loss/n:.4f}  cls={total_cls/n:.4f}  inv={total_inv/n:.4f}  "
              f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
              f"(best_val={best_val_acc:.3f}@{best_epoch}, no_improve={epochs_no_improve}/{args.patience})",
              flush=True)

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"[invreg] Early stopping at epoch {epoch}/{args.epochs} "
                  f"-- no val-acc improvement > {args.min_delta} for {args.patience} epochs "
                  f"(best_val={best_val_acc:.3f}@epoch {best_epoch})", flush=True)
            break

    occ_pool.shutdown(wait=True)

    elapsed = time.time() - _t_start
    print(f"\n[invreg] Best val acc : {best_val_acc:.3f} (epoch {best_epoch})")
    print(f"[invreg] Checkpoint: {ckpt}")
    print(f"[invreg] Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Invariance-regularized (invreg) training")
    p.add_argument("--arch", required=True, choices=list(REGISTRY))
    p.add_argument("--domain", default="modelnet40",
                   choices=["modelnet40", "shapenet", "abo", "s3dis", "scanobjectnn"])
    p.add_argument("--gallery", default=None,
                   help="Directory of *_gallery.ply files (default: the --domain gallery "
                        "under PCROBUST_HOME).")
    p.add_argument("--out", default=None,
                   help="Checkpoint directory (default: the zoo cache location for --domain).")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--micro_batch_size", type=int, default=None,
                   help="Per-forward-pass batch for gradient accumulation (bounds peak "
                        "memory); --batch_size stays the effective batch.")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_inv", type=float, default=0.5)
    p.add_argument("--lambda_inv_warmup_epochs", type=int, default=0,
                   help="Ramp lambda_inv linearly from 0 over this many epochs "
                        "(the paper's _invreg_warmup checkpoints use 10).")
    p.add_argument("--augment", default="default", choices=list(_AUGMENT_MODULES),
                   help="Training-view augmentation set.")
    p.add_argument("--tnet_weight", type=float, default=None,
                   help="Override the T-Net orthogonality weight (default: 0.001 for "
                        "the plain PointNet recipe, else 0).")
    p.add_argument("--occ_subbatch", type=int, default=4)
    p.add_argument("--patience", type=int, default=0,
                   help="Early-stopping patience in epochs (0 = run all --epochs).")
    p.add_argument("--min_delta", type=float, default=0.001)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p


def main(argv=None):
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
