"""Smoke tests: imports, models, checkpoint loading, corruptions, paper data, CLI."""

import importlib
import os
import pkgutil
import sys

import numpy as np
import pytest
import torch

import pcrobust
from pcrobust import paper
from pcrobust.cli import corrupt_points, main as cli_main
from pcrobust.corruptions import add_noise, occlude, random_direction, rotate_yaw
from pcrobust.corruptions.severity import ANGLES, LEVELS, SIGMAS
from pcrobust.models import REGISTRY, get_model
from pcrobust.models.loading import base_arch, load_model
from pcrobust.training.invariance import arch_field, checkpoint_key


def _cloud(n=1024, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, (n, 3)).astype(np.float32)


# ── imports ───────────────────────────────────────────────────────────────────

def test_every_module_imports_without_leaving_the_package():
    # Anything loaded from the directory around this repo (but outside it)
    # would be a broken import on every other machine.
    repo = os.path.dirname(os.path.dirname(os.path.dirname(pcrobust.__file__)))
    outside = os.path.dirname(repo)
    for m in pkgutil.walk_packages(pcrobust.__path__, "pcrobust."):
        importlib.import_module(m.name)
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None) or ""
        assert not (f.startswith(outside) and not f.startswith(repo)), f


def test_import_is_light():
    code = ("import sys, pcrobust; pcrobust.list_models(); "
            "print('torch' in sys.modules, 'matplotlib' in sys.modules)")
    out = os.popen(f'"{sys.executable}" -c "{code}"').read().split()
    assert out == ["False", "False"]


# ── models ────────────────────────────────────────────────────────────────────

def test_registry_matches_metadata():
    assert set(REGISTRY) == set(pcrobust.list_models())
    assert {pcrobust.family_of(k) for k in REGISTRY} == set(pcrobust.FAMILIES)


@pytest.mark.parametrize("arch", list(REGISTRY))
def test_forward_pass_cpu(arch):
    torch.manual_seed(0)
    model = get_model(arch, num_classes=10).eval()
    with torch.no_grad():
        logits, emb, _ = model(torch.randn(2, 3, 1024))
    assert logits.shape == (2, 10)
    assert emb.shape == (2, 128)
    assert torch.allclose(emb.norm(dim=1), torch.ones(2), atol=1e-4)


@pytest.mark.parametrize("key, arch", [
    ("pointnet", "pointnet"),
    ("pointnetpp_invreg", "pointnetpp"),
    ("pointnet_invreg_warmup", "pointnet"),
    ("vn_pointnet_invreg", "vn_pointnet"),
    ("riconvpp_invreg_modelnet40b", "riconvpp"),
    ("dgcnn_invreg_pose270yaw", "dgcnn"),
])
def test_base_arch(key, arch):
    assert base_arch(key) == arch


def test_base_arch_rejects_unknown():
    with pytest.raises(KeyError):
        base_arch("not_a_model")


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = get_model("pointnet", num_classes=10).eval()
    path = tmp_path / "pointnet_invreg.pth"
    classes = [f"c{i}" for i in range(10)]
    torch.save({"arch": "pointnet_invreg", "state_dict": model.state_dict(),
                "classes": classes, "embed_dim": 128, "num_classes": 10}, path)
    loaded, loaded_classes = load_model(str(path), torch.device("cpu"))
    x = torch.randn(1, 3, 1024)
    with torch.no_grad():
        assert torch.allclose(model(x)[0], loaded(x)[0])
    assert loaded_classes == classes


@pytest.mark.parametrize("arch, warmup, augment, domain, key, field", [
    ("pointnet", 0, "default", "modelnet40", "pointnet_invreg", "pointnet_invreg"),
    ("risurconv", 10, "default", "modelnet40", "risurconv_invreg_warmup", "risurconv_invreg_warmup"),
    ("dgcnn", 0, "pose270yaw", "modelnet40", "dgcnn_invreg_pose270yaw", "dgcnn_invreg_pose270yaw"),
    ("riconv", 0, "default", "shapenet", "riconv_invreg", "riconv_invreg_shapenet"),
])
def test_invreg_checkpoint_keys(arch, warmup, augment, domain, key, field):
    assert checkpoint_key(arch, warmup, augment) == key
    assert arch_field(key, domain) == field


# ── corruptions ───────────────────────────────────────────────────────────────

def test_severity_ladders_are_the_papers():
    assert list(ANGLES) == [20, 40, 60, 80, 120]
    assert list(SIGMAS) == [0.005, 0.01, 0.02, 0.03, 0.05]
    assert list(LEVELS) == [0.3, 0.6, 1.2, 2.5, 6.0]


def test_rotation_is_rigid():
    pts = _cloud()
    rot = rotate_yaw(pts, 80)
    np.testing.assert_allclose(np.linalg.norm(rot, axis=1), np.linalg.norm(pts, axis=1), atol=1e-5)
    np.testing.assert_allclose(rotate_yaw(rot, -80), pts, atol=1e-5)


def test_noise_scale():
    np.random.seed(0)
    pts = _cloud(20000)
    assert abs(np.std(add_noise(pts, 0.05) - pts) - 0.05) < 2e-3


def test_occlusion_removes_more_from_closer_camera():
    pts = _cloud()
    d = random_direction(np.random.default_rng(0))
    kept = [len(occlude(pts, level, d)) for level in LEVELS]
    assert all(k <= len(pts) for k in kept)
    assert kept[0] != kept[-1]


@pytest.mark.parametrize("corruption", ["pose", "noise", "occlusion", "rotation"])
def test_corrupt_is_deterministic(corruption):
    pts = _cloud()
    a = corrupt_points(pts, corruption, 3, seed=7)
    b = corrupt_points(pts, corruption, 3, seed=7)
    np.testing.assert_array_equal(a, b)


# ── paper data ────────────────────────────────────────────────────────────────

def test_every_frozen_file_loads():
    names = paper.list_data()
    assert len(names) > 100
    for name in names:
        if name.endswith(".json"):
            assert paper.load_json(name)
        else:
            assert len(paper.load(name)) > 0, name


def test_golden_pose_primary_test():
    """Anchors the frozen data to a number reported in the paper."""
    df = paper.load("summary/pose_primary_test.csv")
    row = df[(df.Domain == "modelnet40") & (df.Architecture == "pointnet")
             & (df.Rotation_Type == "yaw")].iloc[0]
    assert row.n == 740
    assert row.mean_D == pytest.approx(0.41661)
    assert row.Decision == "REJECT H0"


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_list(capsys):
    assert cli_main(["list", "models"]) == 0
    assert "risurconv" in capsys.readouterr().out


def test_cli_corrupt_round_trip(tmp_path):
    src, dst = tmp_path / "in.npy", tmp_path / "out.npy"
    np.save(src, _cloud())
    assert cli_main(["corrupt", str(src), "--corruption", "occlusion",
                     "--severity", "5", "-o", str(dst)]) == 0
    out = np.load(dst)
    assert out.shape[1] == 3 and len(out) < 1024


def test_cli_pass_through_help(capsys):
    assert cli_main(["evaluate", "--help"]) == 0
    assert "pcrobust.evaluate.robustness" in capsys.readouterr().out


# ── checkpoint zoo manifest ───────────────────────────────────────────────────

def test_zoo_manifest_entries_are_well_formed():
    import re

    from pcrobust.cli import DOMAINS
    from pcrobust.models.zoo import _load_manifest, list_checkpoints

    files = _load_manifest()["files"]
    assert len(files) == len(list_checkpoints()) == 193
    for name, entry in files.items():
        domain, fname = name.split("/")
        assert domain in DOMAINS and fname.endswith(".pth"), name
        base_arch(fname[:-4])  # every key resolves to one of the 10 architectures
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), name
        assert entry["bytes"] > 0


def test_zoo_uses_cached_file_when_hash_matches(tmp_path, monkeypatch):
    import hashlib
    import json

    from pcrobust.models import zoo

    data = b"not really a checkpoint"
    manifest = {"revision": "wacv2027", "base_url": None, "files": {
        "modelnet40/pointnet.pth": {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}}}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    monkeypatch.setattr(zoo, "_MANIFEST_PATH", mpath)
    cached = zoo.cache_root() / "checkpoints" / "wacv2027" / "modelnet40" / "pointnet.pth"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(data)
    assert zoo.checkpoint_path("modelnet40/pointnet.pth") == cached


def test_zoo_downloads_flat_asset_name(tmp_path, monkeypatch):
    import hashlib
    import json

    from pcrobust.models import zoo

    data = b"not really a checkpoint"
    manifest = {"revision": "wacv2027", "base_url": "https://example.org/dl", "asset_sep": "__",
                "files": {"modelnet40/pointnet.pth": {
                    "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}}}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    monkeypatch.setattr(zoo, "_MANIFEST_PATH", mpath)
    monkeypatch.setenv("PCROBUST_HOME", str(tmp_path / "home"))
    urls = []

    def fake_urlretrieve(url, dest):
        urls.append(url)
        with open(dest, "wb") as f:
            f.write(data)

    monkeypatch.setattr(zoo.urllib.request, "urlretrieve", fake_urlretrieve)
    path = zoo.checkpoint_path("modelnet40/pointnet.pth")
    assert urls == ["https://example.org/dl/modelnet40__pointnet.pth"]
    assert path == tmp_path / "home" / "checkpoints" / "wacv2027" / "modelnet40" / "pointnet.pth"
    assert path.read_bytes() == data


# ── welcome message ───────────────────────────────────────────────────────────

def test_welcome_message_is_shown_once(capsys):
    from pcrobust._welcome import show_once

    show_once("9.9.9")
    first = capsys.readouterr()
    assert "pcrobust 9.9.9" in first.err and first.out == ""
    show_once("9.9.9")
    assert capsys.readouterr().err == ""


def test_welcome_message_respects_quiet(capsys, monkeypatch):
    from pcrobust._welcome import show_once

    monkeypatch.setenv("PCROBUST_QUIET", "1")
    show_once("9.9.9")
    assert capsys.readouterr().err == ""
