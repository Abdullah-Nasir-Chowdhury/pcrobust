import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point PCROBUST_HOME at a temp dir so tests never touch a real cache."""
    monkeypatch.setenv("PCROBUST_HOME", str(tmp_path / "pcrobust_home"))
