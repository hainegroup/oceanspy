import shutil

import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    """
    Per-test temp directory for on-disk caches (sqlite/filesystem).
    Each test gets its own path; pytest cleans tmp_path automatically.
    """
    d = tmp_path
    d.mkdir(parents=True, exist_ok=True)
    yield d
    # Best-effort cleanup of WAL/SHM etc. (pytest will remove tmp_path anyway)
    shutil.rmtree(d, ignore_errors=True)
