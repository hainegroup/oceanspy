import os
import pathlib

import pooch
from pooch import Untar

DATA_URL = "https://zenodo.org/record/10779221/files/Data.tar.gz?download=1"
DATA_HASH = "md5:b0277b809840e08a84f3a5d2c3f7404b"
EXTRACT_DIRNAME = "oceanspy-test-data"  # stable extracted folder name


def _ensure_symlink(src: str, dst: str) -> None:
    src_path = pathlib.Path(src).resolve()
    dst_path = pathlib.Path(dst)

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # If it's already the correct symlink, do nothing
    if dst_path.is_symlink() and dst_path.resolve() == src_path:
        return

    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()

    os.symlink(str(src_path), str(dst_path), target_is_directory=True)


def _get_data_dir() -> str:
    # CI (or user override): use pre-primed directory
    env_dir = os.environ.get("OCEANSPY_TEST_DATA_DIR")
    if env_dir:
        p = pathlib.Path(env_dir)
        if not p.exists():
            raise RuntimeError(f"OCEANSPY_TEST_DATA_DIR set but missing: {env_dir}")
        return str(p.resolve())

    # Local/dev: download+untar using pooch if needed (uses local cache)
    cache_dir = os.environ.get("POOCH_CACHE_DIR", pooch.os_cache("oceanspy"))
    fnames = pooch.retrieve(
        url=DATA_URL,
        known_hash=DATA_HASH,
        path=cache_dir,
        processor=Untar(extract_dir=EXTRACT_DIRNAME),
    )
    return os.path.commonpath(fnames)


def pytest_sessionstart(session):
    # xdist workers should not do this
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return

    data_dir = _get_data_dir()
    _ensure_symlink(data_dir, "./oceanspy/tests/Data")
