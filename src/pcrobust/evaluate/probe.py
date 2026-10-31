"""
Head-independent probe, stage 1 of 2: embedding extraction.

Purpose: the paper's accuracy numbers all flow through each model's own
trained linear classifier, so a reviewer can object that the invariance
loss improved the CLASSIFIER rather than the REPRESENTATION. This script
extracts frozen embeddings so stage 2 (probe_eval.py) can re-classify the
exact same inputs with training-free heads (subspace/CLAFIC, kNN, nearest
class mean). Agreement across heads is what licenses a claim about the
embedding itself.

Protocol is deliberately identical to
pcrobust.evaluate.accuracy -- same held-out
data/test/ split, same conditions and severities, same K=5 replicate
averaging, same seeds -- so probe accuracy and trained-head accuracy are
directly comparable on identical inputs. The trained-head prediction is
recorded in this same pass for exactly that reason: one forward pass, four
heads, no protocol drift between them.

Every backbone in this project ends in
    emb = F.normalize(self.embed_head(f), dim=1)
so a forward hook on `embed_head` yields the pre-normalization 128-d
vector uniformly across all 8 architectures, with no per-arch special
casing. That gives stage 2 both the normalized embedding (what the
invariance loss actually optimizes) and the unnormalized one (needed for
the raw-Euclidean control that breaks the "you evaluated in the space the
loss optimizes" circularity objection).

Reference vectors come from data/train/ (45 instances per class), queries
from data/test/ (30 per class). Reference and query splits are disjoint.

Usage:
  python -m pcrobust.evaluate.probe --arch riconvpp_invreg --device cuda
  python -m pcrobust.evaluate.probe --arch riconvpp --device cuda

Output: <PCROBUST_HOME>/results/probe/<domain>/emb_<arch>.npz (or --out_dir)
Writes nothing else. Does not touch any existing result, checkpoint, or
script.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pcrobust.corruptions.occlusion import LEVELS, occlude, random_direction
from pcrobust.datasets import checkpoint_file, domain_module, results_dir
from pcrobust.models import get_model
from pcrobust.models.loading import base_arch
from pcrobust.utils import _sample

_rng = np.random.default_rng(42)

ANGLES = [20, 40, 60, 80, 120]
SIGMAS = [0.005, 0.01, 0.02, 0.03, 0.05]
K_REPLICATES = 5
N_POINTS = 1024


def rotate_yaw(pts, deg):
    t = deg * np.pi / 180.0
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return pts @ R.T


def add_noise(pts, sigma):
    return pts + np.random.randn(*pts.shape).astype(np.float32) * sigma


def build_model(ckpt_path, device):
    """Like pcrobust.models.loading.load_model, but also returns the
    checkpoint's own ``arch`` field."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "pointnet")
    model = get_model(base_arch(arch), num_classes=ckpt["num_classes"],
                      embed_dim=ckpt["embed_dim"]).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"], arch


def attach_prenorm_hook(model):
    """Capture embed_head's output, i.e. the embedding BEFORE
    F.normalize. Uniform across all 8 architectures."""
    store = {}

    def _hook(_module, _inp, out):
        store["prenorm"] = out.detach()

    head = getattr(model, "embed_head", None)
    if head is None:
        head = getattr(model, "proj", None)
    if head is None:
        raise RuntimeError("No embed_head/proj layer found, cannot capture "
                           "pre-normalization features for this model.")
    handle = head.register_forward_hook(_hook)
    return store, handle


@torch.no_grad()
def embed_batch(model, pts, device, sample_fn, store, k):
    """Draw k sampling replicates of the SAME input cloud and run them as a
    single batched forward. Returns (emb, prenorm, probs) each with a
    leading k axis.

    This is numerically identical to k separate forward passes: the models
    are in eval() mode so BatchNorm uses running statistics, and every
    operation in these backbones is per-sample. It is much faster because
    the vendored farthest-point sampling loops over POINTS, not over batch,
    so a batch of k costs barely more than a batch of 1. That matters for
    RISurConv and the RIConv family, where that Python loop dominates
    runtime. The replicate draws happen in the same order as the unbatched
    version, so the RNG stream and therefore the resulting embeddings are
    unchanged."""
    ps = np.stack([sample_fn(pts.copy(), N_POINTS) for _ in range(k)])
    x = torch.from_numpy(ps).transpose(1, 2).to(device)
    logits, emb, _ = model(x)
    probs = F.softmax(logits, dim=1)
    return (emb.cpu().numpy(),
            store["prenorm"].cpu().numpy(),
            probs.cpu().numpy())


@torch.no_grad()
def embed_once(model, pts, device, sample_fn, store):
    """One deterministic-input forward pass. Returns (normalized_emb,
    prenorm_emb, softmax_probs), all numpy, all 1-D except probs."""
    p = sample_fn(pts.copy(), N_POINTS)
    x = torch.from_numpy(p).unsqueeze(0).transpose(1, 2).to(device)
    logits, emb, _ = model(x)
    probs = F.softmax(logits, dim=1)
    return (emb.squeeze(0).cpu().numpy(),
            store["prenorm"].squeeze(0).cpu().numpy(),
            probs.squeeze(0).cpu().numpy())


def embed_replicated(model, pts, device, sample_fn, store, k=K_REPLICATES):
    """K replicates, averaged. Softmax is averaged for the trained head
    exactly as classification_accuracy.py does. The normalized embedding is
    averaged then re-normalized, which is the embedding-space analog of
    that averaging, and the pre-normalization vector is averaged plainly.
    Using one shared replicate set keeps all four heads on identical
    inputs."""
    embs, pres, probs = embed_batch(model, pts, device, sample_fn, store, k)
    emb = np.mean(embs, axis=0)
    n = np.linalg.norm(emb)
    emb = emb / n if n > 1e-12 else emb
    return emb, np.mean(pres, axis=0), np.mean(probs, axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract frozen embeddings for "
                                             "the head-independent probe.")
    ap.add_argument("--arch", required=True,
                    help="Checkpoint stem inside --models_dir, e.g. "
                         "riconvpp or riconvpp_invreg.")
    ap.add_argument("--models_dir", default=None,
                    help="Directory holding <arch>.pth (default: the zoo cache "
                         "location for --domain).")
    ap.add_argument("--out_dir", default=None,
                    help="Default: <PCROBUST_HOME>/results/probe/<domain>.")
    ap.add_argument("--domain", default="modelnet40")
    ap.add_argument("--ref_replicates", type=int, default=5,
                    help="Sampling replicates per REFERENCE instance, each "
                         "kept as its own vector. 45 train instances x this "
                         "= vectors per class available to span the class "
                         "subspace.")
    ap.add_argument("--out_tag", default="",
                    help="Appended to the output filename.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    global _rng
    np.random.seed(42)
    torch.manual_seed(42)
    _rng = np.random.default_rng(42)

    dom = domain_module(args.domain)
    extract_object_pts = dom.extract_object_pts
    ckpt = (Path(args.models_dir) / f"{args.arch}.pth" if args.models_dir
            else checkpoint_file(args.arch, args.domain))
    if not ckpt.exists():
        sys.exit(f"[probe_extract] checkpoint not found: {ckpt}")

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, classes, ckpt_arch = build_model(str(ckpt), device)
    store, handle = attach_prenorm_hook(model)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"[probe_extract] arch={args.arch} (ckpt arch={ckpt_arch}) "
          f"device={device} classes={len(classes)}", flush=True)

    def list_split(split):
        d = dom.ROOT / "data" / split
        out = []
        for class_dir in sorted(d.iterdir()):
            if not class_dir.is_dir():
                continue
            for f in sorted(class_dir.glob("*.ply")):
                out.append((class_dir.name, f))
        return out

    # ---------------- reference (train split) ----------------
    ref_files = list_split("train")
    print(f"[probe_extract] reference instances: {len(ref_files)} "
          f"x {args.ref_replicates} replicates", flush=True)

    ref_emb, ref_pre, ref_lab = [], [], []
    for i, (true_class, path) in enumerate(ref_files, 1):
        try:
            pts = extract_object_pts(str(path))
        except Exception as e:
            print(f"[probe_extract] EXTRACTION FAILED {path.name}: {e} SKIPPING",
                  flush=True)
            continue
        es, prs, _probs = embed_batch(model, pts, device, _sample, store,
                                      args.ref_replicates)
        for j in range(args.ref_replicates):
            ref_emb.append(es[j])
            ref_pre.append(prs[j])
            ref_lab.append(cls_to_idx[true_class])
        if i % 50 == 0:
            print(f"[probe_extract]   ref ({i}/{len(ref_files)})", flush=True)

    # ---------------- queries (test split) ----------------
    q_files = list_split("test")
    print(f"[probe_extract] query instances: {len(q_files)}", flush=True)

    q_emb, q_pre, q_lab, q_cond, q_lvl, q_head = [], [], [], [], [], []

    def record(pts, cond, lvl, true_class):
        e, p, probs = embed_replicated(model, pts, device, _sample, store)
        q_emb.append(e)
        q_pre.append(p)
        q_lab.append(cls_to_idx[true_class])
        q_cond.append(cond)
        q_lvl.append(str(lvl))
        q_head.append(int(np.argmax(probs)))

    for i, (true_class, path) in enumerate(q_files, 1):
        try:
            pts = extract_object_pts(str(path))
        except Exception as e:
            print(f"[probe_extract] EXTRACTION FAILED {path.name}: {e} SKIPPING",
                  flush=True)
            continue

        record(pts, "clean", "", true_class)
        for angle in ANGLES:
            record(rotate_yaw(pts, angle), "pose", angle, true_class)
        for sigma in SIGMAS:
            record(add_noise(pts, sigma), "noise", sigma, true_class)
        direction = random_direction(_rng)
        for dist_mult in LEVELS:
            record(occlude(pts, dist_mult, direction), "occlusion",
                   dist_mult, true_class)

        if i % 25 == 0:
            print(f"[probe_extract]   query ({i}/{len(q_files)})", flush=True)

    handle.remove()

    out_dir = Path(args.out_dir) if args.out_dir else results_dir("probe", args.domain)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"emb_{args.arch}{args.out_tag}.npz"
    np.savez_compressed(
        out_path,
        ref_emb=np.asarray(ref_emb, dtype=np.float32),
        ref_prenorm=np.asarray(ref_pre, dtype=np.float32),
        ref_label=np.asarray(ref_lab, dtype=np.int64),
        q_emb=np.asarray(q_emb, dtype=np.float32),
        q_prenorm=np.asarray(q_pre, dtype=np.float32),
        q_label=np.asarray(q_lab, dtype=np.int64),
        q_condition=np.asarray(q_cond),
        q_level=np.asarray(q_lvl),
        q_head_pred=np.asarray(q_head, dtype=np.int64),
        classes=np.asarray(classes),
        arch=np.asarray([args.arch]),
    )
    print(f"[probe_extract] saved {out_path}  "
          f"ref={len(ref_emb)} queries={len(q_emb)}", flush=True)


if __name__ == "__main__":
    main()
