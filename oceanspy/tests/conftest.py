# Import modules
import os

import pooch
from pooch import Untar


# Download data if necessary
def pytest_configure():
    fnames = pooch.retrieve(
        url="https://zenodo.org/record/10713953/files/Data.tar.gz?download=1",
        processor=Untar(),
        known_hash="md5:92a2e858a6ba1c37d37fedbd5e932797",
    )
    symlink_args = dict(
        src=f"{os.path.commonpath(fnames)}",
        dst="./oceanspy/tests/Data",
        target_is_directory=True,
    )
    try:
        print(f"Linking {symlink_args['src']!r} to {symlink_args['dst']!r}")
        os.symlink(**symlink_args)
    except FileExistsError:
        os.unlink("./oceanspy/tests/Data")
        os.symlink(**symlink_args)
