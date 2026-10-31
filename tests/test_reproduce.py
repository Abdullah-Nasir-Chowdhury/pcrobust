"""Golden tests: the bundled frozen data rebuilds the paper's tables and figures."""

import json

import pandas as pd
import pytest

from pcrobust.cli import main as cli_main
from pcrobust.paper import reproduce
from pcrobust.paper.latex import clean_cell, matches, numbers, parse_tabular

TARGETS = [t.id for t in reproduce.list_targets()]

# Cells where the printed number is not a correct rounding of the data, all
# off by one in the third decimal. Each comes from rounding twice (the
# value was first stored at 4 decimals, then printed at 3) or from
# averaging already-rounded values:
#   PointNet++ (loss) corrupted acc  0.80047 -> printed 0.801, gap 0.092
#   PointTransf. (base) corrupted    0.77747 -> printed 0.778, gap 0.153
#   seeds: RIConv++ corr 0.79050, RISurConv mCE 1.92250, RIConv (loss) mCE sd 0.15448
#   DiD "Mean, RI" occlusion          0.11863 -> printed 0.118
# When the paper is corrected, re-run tools/sync_paper.py and empty these.
KNOWN_PAPER_ROUNDING = {
    "tab:m40c": {("PointTransf.|no", 1), ("PointTransf.|no", 3),
                 ("PointNet++|yes", 1), ("PointNet++|yes", 3)},
    "tab:supp-w4": {("PointNet++", 5), ("PointTransf.", 3)},
    "tab:supp-seeds": {("RIConv++|no", 1), ("RISurConv|no", 2), ("RIConv|yes", 2)},
    "tab:supp-did": {("Mean, RI", 3)},
}


# ── golden: every printed number ──────────────────────────────────────────────

@pytest.mark.parametrize("target", TARGETS)
def test_target_reproduces_the_paper(target):
    rep = reproduce.compare(target)
    assert rep.checked > 0
    assert {(k, c) for k, c, *_ in rep.mismatches} == KNOWN_PAPER_ROUNDING.get(target, set()), \
        rep.mismatches
    if rep.uncovered:
        assert reproduce.get_target(target).missing, rep.uncovered


def test_every_paper_table_has_a_target():
    snapshot = {p.name for p in (reproduce._expected_root()).iterdir()
                if p.name.endswith(".csv")}
    assert snapshot == {t.replace(":", "_") + ".csv" for t in TARGETS}


def test_snapshot_records_only_the_commit():
    src = reproduce.paper_source()
    assert set(src) == {"commit", "commit_date", "uncommitted_changes"}
    assert len(src["commit"]) == 40
    assert "\\" not in json.dumps(src) and "/" not in src["commit"]


# ── golden: anchors that do not depend on the snapshot ────────────────────────

@pytest.mark.parametrize("target, row, col, value", [
    ("tab:noise", "RIConv++", "modelnet40", 0.537),
    ("tab:noise", "VN-DGCNN", "s3dis", 0.626),
    ("tab:m40c", "RIConv++|yes", "mCE", 0.950),
    ("tab:m40c", "PointMLP|yes", "corrupted", 0.823),
    ("tab:transfer", "lidar", "delta_acc", 0.134),
    ("tab:transfer", "background", "base_acc", 0.299),
    ("tab:rotation_gap", "PointMLP", "mean_gap", 0.037),
    ("tab:rotation_gap", "PointNet++", "widest_gap", 0.133),
    ("tab:domains", "S3DIS", "confirm_pool", 217),
])
def test_headline_numbers(target, row, col, value):
    assert reproduce.build(target).loc[row, col] == pytest.approx(value, abs=5e-4)


def test_ci_cells_are_mean_low_high():
    cell = reproduce.build("tab:supp-noise").loc["PointNet", "modelnet40"]
    assert cell == pytest.approx((0.143, 0.130, 0.157), abs=5e-4)


def test_partial_targets_say_what_is_missing():
    df = reproduce.build("tab:supp-cost")
    assert df["latency_ms"].isna().all() and df["params_M"].notna().all()
    assert "latency" in reproduce.get_target("tab:supp-cost").missing


def test_lookup_by_bare_name():
    assert reproduce.get_target("noise").id == "tab:noise"
    assert reproduce.get_target("teaser").id == "fig:teaser"
    with pytest.raises(KeyError):
        reproduce.get_target("tab:nope")


def test_compare_flags_a_wrong_number():
    df = reproduce.build("tab:noise")
    df.loc["PointNet", "modelnet40"] = 0.2
    rep = reproduce.compare("tab:noise", df)
    assert [(k, c) for k, c, *_ in rep.mismatches] == [("PointNet", 0)]


# ── LaTeX parsing ─────────────────────────────────────────────────────────────

def test_clean_cell():
    assert clean_cell(r"\textbf{+46.3}") == "+46.3"
    assert clean_cell(r"$-$0.013") == "-0.013"
    assert clean_cell(r"0.001 [-0.002, 0.005]$^{\text{ns}}$") == "0.001 [-0.002, 0.005]"
    assert clean_cell(r"$+$0.051$^{*}$") == "+0.051"
    assert clean_cell(r".90$\rightarrow$.58") == ".90->.58"
    assert clean_cell(r"density\_inc") == "density_inc"


def test_numbers_and_rounding():
    assert numbers("0.143 [0.130, 0.157]") == ["0.143", "0.130", "0.157"]
    assert numbers("+0.170 (9)") == ["+0.170", "9"]
    assert numbers("0.928 +/- 0.026") == ["0.928", "0.026"]
    assert numbers("[-.036, .002]") == ["-.036", ".002"]
    assert matches("0.143", 0.14332) and matches("0.143", 0.1435)
    assert not matches("0.143", 0.1436)
    assert matches("+19.7", 19.66) and matches("-0.000", -0.0003)


def test_parse_tabular_keys():
    body = r"""Model & Head & A \\ \midrule
PointNet & trained & 0.890 \\
 & kNN & 0.927 \\
\midrule
\multicolumn{3}{l}{\emph{group}} \\
DiD & x & $+$0.051 \\
\quad 95\% CI & x & [.026, .075] \\
"""
    rows = parse_tabular(body, key_cols=2)
    assert [k for k, _ in rows] == ["PointNet|trained", "PointNet|kNN", "DiD|x", "DiD|x|95% CI|x"]
    assert rows[1][1] == ["0.927"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_list(capsys):
    assert cli_main(["reproduce", "--list"]) == 0
    out = capsys.readouterr().out
    assert "tab:noise" in out and "fig:rotation_sweep" in out and "partial:" in out


def test_cli_compare_one_target(capsys):
    assert cli_main(["reproduce", "noise", "--compare-paper"]) == 0
    out = capsys.readouterr().out
    assert "0.537" in out and "50/50 printed numbers reproduced" in out


def test_cli_compare_reports_differences(capsys):
    assert cli_main(["reproduce", "tab:m40c", "--compare-paper"]) == 1
    assert "paper 0.801, rebuilt 0.80047" in capsys.readouterr().out


def test_cli_out_writes_csv(tmp_path):
    assert cli_main(["reproduce", "tab:supp-noise", "tab:domains", "--out", str(tmp_path)]) == 0
    df = pd.read_csv(tmp_path / "tab_supp-noise.csv", index_col="row")
    assert list(df.columns[:3]) == ["modelnet40", "modelnet40_lo", "modelnet40_hi"]
    assert (tmp_path / "tab_domains.csv").is_file()


def test_cli_figure(tmp_path):
    pytest.importorskip("matplotlib")
    assert cli_main(["reproduce", "fig:teaser", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "fig_teaser.png").stat().st_size > 0


def test_cli_unknown_target(capsys):
    assert cli_main(["reproduce", "tab:nope"]) == 2
