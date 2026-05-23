import pytest

from blscint.ne2001 import ne2001


def test_query_ne2001_reports_missing_executable(monkeypatch):
    real_exists = ne2001.os.path.exists

    def fake_exists(path):
        if path.endswith("bin.NE2001/NE2001"):
            return False
        return real_exists(path)

    monkeypatch.setattr(ne2001.os.path, "exists", fake_exists)

    with pytest.raises(FileNotFoundError, match="make pgm"):
        ne2001.query_ne2001(0, 0, 8.2, field="SCINTIME")
