"""Training-gallery dataset: normalized ``*_gallery.ply`` point clouds."""

import numpy as np
import torch
import open3d as o3d
from pathlib import Path
from torch.utils.data import Dataset

from pcrobust.utils import _augment, _sample


class MeshGalleryDataset(Dataset):
    """
    Loads *_gallery.ply files from gallery_dir.
    Category parsed from stem: airplane1_gallery → airplane.
    Files with purely numeric stems (from data/test/) are skipped.
    """

    def __init__(self, gallery_dir: str, n_points: int = 1024,
                 augment: bool = True, repeats: int = 100):
        gallery_dir = Path(gallery_dir)
        ply_files = sorted(gallery_dir.glob("*_gallery.ply"))
        if not ply_files:
            raise FileNotFoundError(
                f"No *_gallery.ply files found in {gallery_dir}\n"
                f"Run the relevant domain's prepare_dataset() first "
                f"(e.g. pcrobust.datasets.modelnet40.prepare_dataset())"
            )

        self.n_points   = n_points
        self.do_augment = augment
        self.repeats    = repeats if augment else 1

        raw, categories = [], []
        for f in ply_files:
            name = f.stem.replace("_gallery", "").rstrip("0123456789")
            if not name:
                continue
            pcd = o3d.io.read_point_cloud(str(f))
            pts = np.asarray(pcd.points, dtype=np.float32)
            raw.append(pts)
            categories.append(name)

        self.classes      = sorted(set(categories))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.raw          = raw
        self.labels       = [self.class_to_idx[n] for n in categories]

        print(f"[data]  {len(self.classes)} categories  |  {len(raw)} scans  |  "
              f"{len(self):,} samples ({repeats}x aug)")

    def __len__(self):
        return len(self.raw) * self.repeats

    def __getitem__(self, idx):
        i   = idx % len(self.raw)
        pts = self.raw[i].copy()
        if self.do_augment:
            pts = _augment(pts)
        pts = _sample(pts, self.n_points)
        return torch.from_numpy(pts), self.labels[i]
