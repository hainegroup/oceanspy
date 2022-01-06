# Import modules
import os

import pooch
from pooch import Untar


# Download data if necessary
def pytest_configure():

    fnames = pooch.retrieve(
        url="https://zenodo.org/record/5825166/files/Data.tar.gz?download=1",
        processor=Untar(),
        known_hash="165bb4c0459a3b776efe9f94e8f7843140bc7e90b9d686af574cdb8c11006ba2",
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
