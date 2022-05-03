import dask.array as da
import pytest
from napari.layers import Image, Labels
from qtpy.QtCore import Qt

from tracking_challenge_demo import (
    SegmentationDiffHighlight,
    Threshold,
    segment_by_threshold,
)


@pytest.fixture
def im_layer():
    return Image(da.random.random((5, 100, 100)), name="im")


@pytest.fixture
def labels_layer():
    return Labels(da.random.randint(0, 255, (5, 100, 100)), name="lab")


def test_segment_widg_returns_layer(im_layer, labels_layer):
    widg = segment_by_threshold()

    retval = widg(im_layer, Threshold.triangle)
    assert isinstance(retval[0], da.Array)
    assert retval[1]["name"] == "im_seg"
    assert retval[2] == "labels"


def test_highlight_widg_populates_layers(
    make_napari_viewer, labels_layer, im_layer
):
    viewer = make_napari_viewer()
    widg = SegmentationDiffHighlight(viewer)

    assert widg.gt_layer_combo.count() == 0
    assert widg.seg_layer_combo.count() == 0

    viewer.add_layer(labels_layer)
    viewer.add_layer(im_layer)
    assert widg.gt_layer_combo.count() == 1
    assert widg.gt_layer_combo.currentText() == "lab"
    assert widg.seg_layer_combo.count() == 1
    assert widg.seg_layer_combo.currentText() == "lab"


def test_highlight_widg_computes_difference(
    make_napari_viewer, labels_layer, qtbot
):
    viewer = make_napari_viewer()
    widg = SegmentationDiffHighlight(viewer)
    viewer.add_layer(labels_layer)
    assert len(viewer.layers) == 1

    qtbot.mouseClick(widg.highlight_btn, Qt.MouseButton.LeftButton)

    assert len(viewer.layers) == 2
