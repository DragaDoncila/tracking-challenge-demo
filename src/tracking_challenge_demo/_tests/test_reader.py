import numpy as np
from tifffile import imsave

from tracking_challenge_demo import napari_get_reader


def test_reader_gt(tmpdir):
    """Test gt folders with no sequence are read correctly"""

    # write some fake data using your supported file format
    gt_pth = tmpdir.mkdir("01_GT").mkdir("SEG")

    gt_labels = np.random.randint(0, 20, size=(100, 100), dtype=np.uint8)
    gt_tif = gt_pth.join("man_seg000.tif")
    imsave(str(gt_tif), gt_labels)

    # try to read it back in
    reader = napari_get_reader(gt_pth)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(str(gt_pth))
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 1
    layer_data_tuple = layer_data_list[0]
    assert (
        isinstance(layer_data_tuple, tuple) and layer_data_tuple[2] == "labels"
    )

    # make sure it's the same as it started
    layer_data = layer_data_tuple[0]
    assert layer_data.shape == (1, 100, 100)
    assert all([np.allclose(gt_labels, im_slice) for im_slice in layer_data])


def test_reader_gt_w_seq(tmpdir):
    """Test gt frames with sister sequence are read correctly"""

    seq_pth = tmpdir.mkdir("01")
    original_data = np.random.randint(
        0, 2 ** 12, size=(100, 100), dtype=np.uint16
    )
    seq_tif = seq_pth.join("t000.tif")
    seq2_tif = seq_pth.join("t001.tif")
    imsave(str(seq_tif), original_data)
    imsave(str(seq2_tif), original_data)

    gt_pth = tmpdir.mkdir("01_GT").mkdir("SEG")

    gt_labels = np.random.randint(0, 20, size=(100, 100), dtype=np.uint8)
    gt_tif = gt_pth.join("man_seg001.tif")
    imsave(str(gt_tif), gt_labels)

    # try to read it back in
    reader = napari_get_reader(gt_pth)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(str(gt_pth))
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 1
    layer_data_tuple = layer_data_list[0]
    assert (
        isinstance(layer_data_tuple, tuple) and layer_data_tuple[2] == "labels"
    )

    # make sure it's the same as it started
    layer_data = layer_data_tuple[0]
    assert layer_data.shape == (2, 100, 100)
    np.testing.assert_allclose(gt_labels, layer_data[1])


def test_get_reader_gt_pass(tmpdir):
    """Valid dir of GT tiffs"""
    seq = tmpdir.mkdir("01_GT").mkdir("SEG")
    seq.join("man_seg000.tif").write("test")

    reader = napari_get_reader(seq)
    assert callable(reader)


def test_get_reader_not_dir_fail(tmpdir):
    """Not receiving a directory"""
    seq = tmpdir.mkdir("01").join("t000.tif")
    reader = napari_get_reader(seq)

    assert reader is None


def test_get_reader_wrong_dir_name_fail(tmpdir):
    """Directory incorrectly named"""
    seq = tmpdir.mkdir("foobar").join("t000.tif")
    reader = napari_get_reader(seq)

    assert reader is None


def test_get_reader_mix_tif_fail(tmpdir):
    """Directory ok but tifs are not named by expected sequence"""
    seq = tmpdir.mkdir("01")
    seq.join("t000.tif").write("test")
    seq.join("foobar.tif").write("test")

    reader = napari_get_reader(seq)
    assert reader is None


def test_get_reader_mix_dir_fail(tmpdir):
    """Directory is sequence but files are GT or vice versa"""
    seq = tmpdir.mkdir("01")
    seq.join("man_seg000.tif").write("test")
    reader = napari_get_reader(seq)
    assert reader is None

    seq = tmpdir.mkdir("01_GT")
    seq.join("t000.tif").write("test")
    reader = napari_get_reader(seq)
    assert reader is None


def test_get_reader_no_tif_fail(tmpdir):
    """Directory is ok but no tifs are present"""
    seq = tmpdir.mkdir("01")
    seq.join("not_a_tif.jpg").write("test")

    reader = napari_get_reader(seq)
    assert reader is None
