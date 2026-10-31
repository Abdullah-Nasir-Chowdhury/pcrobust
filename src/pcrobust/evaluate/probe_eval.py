"""
Head-independent probe, stage 2 of 2: training-free
classification of the frozen embeddings produced by pcrobust.evaluate.probe.

Four heads, identical inputs:

  head      the model's own trained linear classifier (recorded during
            extraction, reproduced here for reference only)
  sm        subspace method / CLAFIC. Per-class basis from the SVD of the
            class reference matrix (columns are feature vectors, no mean
            centering, matching the classical formulation), query assigned
            to the class maximizing the squared projection length onto its
            basis (the classical subspace method).
  knn       k-nearest-neighbour on cosine similarity over normalized
            embeddings.
  ncm       nearest class mean, cosine, over normalized embeddings.
  ncm_eucl  nearest class mean, plain Euclidean, over the PRE-normalization
            embeddings. This is the circularity control. sm/knn/ncm all
            operate in the same normalized space the invariance loss
            optimizes, so agreement among them is weaker evidence than it
            looks. ncm_eucl leaves that space entirely.

None of these heads is proposed as a method. They are measurement
instruments for one question: does the invariance loss improve the
representation, or only the classifier trained on top of it?

Usage:
  # one model, sweep subspace dimension
  python -m pcrobust.evaluate.probe_eval \
      --emb <PCROBUST_HOME>/results/probe/modelnet40/emb_riconvpp_invreg.npz \
      --sweep_k

  # all extracted models, fixed k, write the comparison CSV
  python -m pcrobust.evaluate.probe_eval \
      --emb_dir <PCROBUST_HOME>/results/probe/modelnet40 \
      --sm_dim 20 --out probe_summary.csv

Output (with --out): one row per (arch, head, condition, level, accuracy).
Prints a per-model summary either way. Writes nothing else.
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np


def get_subspace(matrix, n_dim):
    """Orthonormal basis for the leading n_dim directions of `matrix`,
    whose COLUMNS are feature vectors. Plain SVD, no mean centering, which
    is the classical subspace-method convention."""
    U, _S, _V = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :min(n_dim, U.shape[1])]


def sm_predict(ref_emb, ref_label, q_emb, n_class, sm_dim):
    """CLAFIC. Similarity of a query to a class is the squared length of
    its projection onto that class subspace."""
    bases = []
    for c in range(n_class):
        M = ref_emb[ref_label == c].T          # (dim, n_samples)
        bases.append(get_subspace(M, sm_dim))
    Y = q_emb.T                                  # (dim, n_queries)
    sim = np.stack([np.sum((B.T @ Y) ** 2, axis=0) for B in bases], axis=0)
    return np.argmax(sim, axis=0)


def knn_predict(ref_emb, ref_label, q_emb, n_class, k=5):
    """Cosine kNN. Inputs are already L2 normalized, so the dot product is
    the cosine."""
    sims = q_emb @ ref_emb.T                     # (n_queries, n_ref)
    idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    preds = np.empty(q_emb.shape[0], dtype=np.int64)
    for i in range(q_emb.shape[0]):
        neigh = ref_label[idx[i]]
        w = sims[i, idx[i]]
        score = np.zeros(n_class)
        for lab, wt in zip(neigh, w):
            score[lab] += wt
        preds[i] = int(np.argmax(score))
    return preds


def ncm_predict(ref_emb, ref_label, q_emb, n_class, metric="cosine"):
    cents = np.stack([ref_emb[ref_label == c].mean(axis=0)
                      for c in range(n_class)], axis=0)
    if metric == "cosine":
        cn = np.linalg.norm(cents, axis=1, keepdims=True)
        cents = cents / np.maximum(cn, 1e-12)
        qn = np.linalg.norm(q_emb, axis=1, keepdims=True)
        q = q_emb / np.maximum(qn, 1e-12)
        return np.argmax(q @ cents.T, axis=1)
    d = ((q_emb[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d, axis=1)


def accuracy_by_condition(preds, labels, conds, levels):
    groups = defaultdict(list)
    for p, l, c, lv in zip(preds, labels, conds, levels):
        groups[(str(c), str(lv))].append(int(p == l))
    out = OrderedDict()
    for key in sorted(groups.keys()):
        v = groups[key]
        out[key] = (len(v), float(np.mean(v)))
    return out


def evaluate_one(path, sm_dim, knn_k, sweep_k):
    d = np.load(path, allow_pickle=True)
    arch = str(d["arch"][0])
    # The npz records the base architecture, which stops being unique once
    # the same arch is extracted for several seeds or training regimes.
    # The filename carries the run tag, so label rows from it instead:
    #   emb_dgcnn_invreg_rep2.npz -> model dgcnn_invreg_rep2, run rep2
    #   emb_pointnet_so3aug.npz   -> model pointnet_so3aug,   run so3aug
    #   emb_dgcnn.npz             -> model dgcnn,             run main
    model = Path(path).stem
    if model.startswith("emb_"):
        model = model[4:]
    run = model[len(arch):].lstrip("_") if model.startswith(arch) else ""
    run = run or "main"
    classes = [str(c) for c in d["classes"]]
    n_class = len(classes)

    ref_emb = d["ref_emb"]
    ref_pre = d["ref_prenorm"]
    ref_label = d["ref_label"]
    q_emb = d["q_emb"]
    q_pre = d["q_prenorm"]
    q_label = d["q_label"]
    q_cond = d["q_condition"]
    q_lvl = d["q_level"]
    q_head = d["q_head_pred"]

    per_class = np.bincount(ref_label, minlength=n_class)
    max_dim = int(per_class.min())
    print(f"\n=== {model} ===")
    print(f"  ref {ref_emb.shape}  queries {q_emb.shape}  "
          f"classes {n_class}  min ref/class {max_dim}")

    if sweep_k:
        print("  subspace-dimension sweep (clean accuracy):")
        for k in [1, 2, 5, 10, 15, 20, 25, 30, 40]:
            if k > max_dim:
                break
            p = sm_predict(ref_emb, ref_label, q_emb, n_class, k)
            m = (q_cond == "clean")
            acc = float(np.mean(p[m] == q_label[m]))
            print(f"    k={k:3d}  clean acc {acc:.4f}")

    eff_dim = min(sm_dim, max_dim)
    if eff_dim != sm_dim:
        print(f"  NOTE: sm_dim {sm_dim} exceeds available {max_dim}, "
              f"using {eff_dim}")

    heads = OrderedDict()
    heads["head"] = q_head
    heads["sm"] = sm_predict(ref_emb, ref_label, q_emb, n_class, eff_dim)
    heads["knn"] = knn_predict(ref_emb, ref_label, q_emb, n_class, knn_k)
    heads["ncm"] = ncm_predict(ref_emb, ref_label, q_emb, n_class, "cosine")
    heads["ncm_eucl"] = ncm_predict(ref_pre, ref_label, q_pre, n_class,
                                    "euclidean")

    rows = []
    table = OrderedDict()
    for name, preds in heads.items():
        table[name] = accuracy_by_condition(preds, q_label, q_cond, q_lvl)
        for (cond, lvl), (n, acc) in table[name].items():
            rows.append({"model": model, "arch": arch, "run": run,
                         "head": name, "condition": cond,
                         "level": lvl, "n": n, "accuracy": round(acc, 6),
                         "sm_dim": eff_dim if name == "sm" else "",
                         "knn_k": knn_k if name == "knn" else ""})

    keys = ["('clean', '')", "('noise', '0.05')", "('occlusion', '6.0')",
            "('pose', '120')"]
    print(f"  {'head':10s} {'clean':>8s} {'noise.05':>9s} "
          f"{'occl6.0':>8s} {'pose120':>8s}")
    for name in heads:
        vals = []
        for k in [("clean", ""), ("noise", "0.05"),
                  ("occlusion", "6.0"), ("pose", "120")]:
            vals.append(table[name].get(k, (0, float("nan")))[1])
        print(f"  {name:10s} {vals[0]:8.3f} {vals[1]:9.3f} "
              f"{vals[2]:8.3f} {vals[3]:8.3f}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", help="Single .npz from pcrobust.evaluate.probe")
    ap.add_argument("--emb_dir", help="Directory of emb_*.npz")
    ap.add_argument("--sm_dim", type=int, default=20)
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--sweep_k", action="store_true",
                    help="Print a subspace-dimension sensitivity sweep.")
    ap.add_argument("--out", default=None, help="CSV path for all rows.")
    args = ap.parse_args(argv)

    if args.emb:
        paths = [Path(args.emb)]
    elif args.emb_dir:
        paths = sorted(Path(args.emb_dir).glob("emb_*.npz"))
    else:
        ap.error("give --emb or --emb_dir")
    if not paths:
        ap.error("no embedding files found")

    all_rows = []
    for p in paths:
        all_rows.extend(evaluate_one(p, args.sm_dim, args.knn_k, args.sweep_k))

    if args.out:
        out = Path(args.out)
        if not out.is_absolute() and args.emb_dir:
            out = Path(args.emb_dir) / out
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["model", "arch", "run",
                                              "head", "condition",
                                              "level", "n", "accuracy",
                                              "sm_dim", "knn_k"])
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n[probe_eval] wrote {out}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
