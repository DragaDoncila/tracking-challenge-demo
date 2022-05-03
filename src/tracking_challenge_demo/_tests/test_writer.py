import os
from zipfile import ZipFile

import numpy as np

from tracking_challenge_demo import write_single_image


def test_writing_single_layer(tmpdir, qtbot):
    layer_data = np.random.randint(0, 255, (5, 100, 100))
    pth = os.path.join(str(tmpdir), "test_labels.zip")
    write_single_image(pth, layer_data, {})

    def check_zip():
        assert os.path.exists(pth)
        zip_file = ZipFile(pth)
        tiff_filenames = [
            info.filename
            for info in zip_file.infolist()
            if ".tif" in info.filename
        ]
        assert len(tiff_filenames) == 5
        assert all(["01_AUTO/SEG/seg00" in fname for fname in tiff_filenames])

    # we use waitUntil to wait for the thread to finish without needing
    # its finished signal
    qtbot.waitUntil(check_zip)


def test_writing_returns_none(tmpdir):
    layer_data = np.random.randint(0, 255, (100, 100))
    pth = os.path.join(str(tmpdir), "test_labels.zip")
    assert write_single_image(pth, layer_data, {}) is None
