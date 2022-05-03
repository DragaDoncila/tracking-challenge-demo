"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/plugins/guides.html?#readers
"""
import glob
import os
import re
import warnings
from pathlib import Path

import dask
import dask.array as da
import tifffile
from napari.utils import progress

SEQ_REGEX = r"(.*)/([0-9]{2,})$"
GT_REGEX = r"(.*)/([0-9]{2,})(_(?:GT|AUTO))/SEG$"

SEQ_TIF_REGEX = rf'{SEQ_REGEX[:-1]}/t([0-9]{{3}}){"."}tif$'
GT_TIF_REGEX = rf'{GT_REGEX[:-1]}/(?:man_)?seg([0-9]{{3}}){"."}tif$'


def napari_get_reader(path):
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return None
    is_gt = re.match(GT_REGEX, path)
    if not is_gt:
        return None
    all_tifs = glob.glob(path + "/*.tif")
    if not all_tifs:
        return None
    is_gt_tifs = all([re.match(GT_TIF_REGEX, pth) for pth in all_tifs])
    if not is_gt_tifs:
        return None

    return reader_function


def reader_function(path):
    path = os.path.normpath(path)
    gt_match = re.match(GT_REGEX, path)

    parent_dir_pth = Path(path).parent.parent.absolute()
    seq_number = gt_match.groups()[1]
    sister_sequence_pth = os.path.join(parent_dir_pth, seq_number)

    n_frames = None
    if not os.path.exists(sister_sequence_pth):
        warnings.warn(
            f"Can't find image for ground truth at {path}. Reading without knowing number of frames..."
        )
    else:
        latest_tif_pth = sorted(glob.glob(sister_sequence_pth + "/*.tif"))[-1]
        n_frames = (
            int(re.match(SEQ_TIF_REGEX, latest_tif_pth).groups()[-1]) + 1
        )

    all_tifs = sorted(
        pth
        for pth in glob.glob(path + "/*.tif")
        if re.match(GT_TIF_REGEX, pth)
    )
    tif_shape = None
    tif_dtype = None
    with tifffile.TiffFile(all_tifs[0]) as im_tif:
        tif_shape = im_tif.pages[0].shape
        tif_dtype = im_tif.pages[0].dtype
    if not n_frames:
        n_frames = len(all_tifs)
    im_stack = [
        da.zeros(shape=tif_shape, dtype=tif_dtype) for _ in range(n_frames)
    ]

    @dask.delayed
    def read_im(tif_pth):
        with tifffile.TiffFile(tif_pth) as im_tif:
            im = im_tif.pages[0].asarray()
        return im

    for tif_pth in progress(all_tifs):
        frame_index = int(re.match(GT_TIF_REGEX, tif_pth).groups()[-1])
        im = da.from_delayed(
            read_im(tif_pth), shape=tif_shape, dtype=tif_dtype
        )
        im_stack[frame_index] = im
    layer_data = da.stack(im_stack)

    layer_type = "labels"
    layer_name = f"{gt_match.group(2)}{gt_match.group(3)}"
    add_kwargs = {"name": layer_name}

    return [(layer_data, add_kwargs, layer_type)]
