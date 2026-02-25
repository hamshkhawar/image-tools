"""Provides the function to apply flatfield."""

import logging
import pathlib
import typing
from concurrent.futures import ProcessPoolExecutor, as_completed

import bfio
import numpy
import numpy as np
import tqdm
from filepattern import FilePattern

from . import utils

logger = logging.getLogger(__name__)
logger.setLevel(utils.POLUS_LOG)


def apply(
    *,
    img_dir: pathlib.Path,
    img_pattern: str,
    ff_dir: pathlib.Path,
    ff_pattern: str,
    df_pattern: typing.Optional[str],
    out_dir: pathlib.Path,
    preview: bool = False,
    keep_orig_dtype: typing.Optional[bool] = True,
) -> typing.List[pathlib.Path]:
    """Run batch-wise flatfield correction on the image collection.

    Args:
        img_dir: path to the directory containing the images to be processed.
        img_pattern: filename pattern used to select images from img_dir.
        ff_dir: path to the directory containing the flatfield images.
        ff_pattern: filename pattern used to select flatfield components from
        ff_dir.
        df_pattern: filename pattern used to select darkfield components from
        ff_dir.
        out_dir: path to the directory where the corrected images will be
        saved.
        preview: if True, return the paths to the images that would be saved
        without actually performing any other computation.
        keep_orig_dtype: if True, the output images will be saved with the same
        dtype as the input images. If False, the output images will be saved as
        float32.
    """
    img_fp = FilePattern(str(img_dir), img_pattern)
    img_variables = img_fp.get_variables()

    ff_fp = FilePattern(str(ff_dir), ff_pattern)
    ff_variables = ff_fp.get_variables()

    if set(ff_variables) - set(img_variables):
        msg = (
            f"Flatfield variables are not a subset of image variables: "
            f"{ff_variables} - {img_variables}"
        )
        logger.error(msg)
        raise ValueError(msg)

    if (df_pattern is None) or (not df_pattern):
        df_fp = None
    else:
        df_fp = FilePattern(str(ff_dir), df_pattern)
        df_variables = df_fp.get_variables()
        if set(df_variables) != set(ff_variables):
            msg = (
                f"Flatfield and darkfield variables do not match: "
                f"{ff_variables} != {df_variables}"
            )
            logger.error(msg)
            raise ValueError(msg)

    out_files = []
    for group, files in img_fp(group_by=ff_variables):
        img_paths = [p for _, [p] in files]
        variables = dict(group)

        ff_path: pathlib.Path = ff_fp.get_matching(**variables)[0][1][0]
        df_path = None if df_fp is None else df_fp.get_matching(**variables)[0][1][0]

        if preview:
            out_files.extend(img_paths)
        else:
            _unshade_images(img_paths, out_dir, ff_path, df_path, keep_orig_dtype)

    return out_files


def _unshade_images(
    img_paths: typing.List[pathlib.Path],
    out_dir: pathlib.Path,
    ff_path: pathlib.Path,
    df_path: typing.Optional[pathlib.Path],
    keep_orig_dtype: typing.Optional[bool] = True,
) -> None:
    """Remove the given flatfield components from all images and save outputs.

    Args:
        img_paths: list of paths to images to be processed
        out_dir: directory to save the corrected images
        ff_path: path to the flatfield image
        df_path: path to the darkfield image
        keep_orig_dtype: if True, the output images will be saved with the same
        dtype as the input images. If False, the output images will be saved as
        float32.
    """
    logger.info(f"Applying flatfield correction to {len(img_paths)} images ...")
    logger.info(f"{ff_path.name = } ...")
    logger.debug(f"Images: {img_paths}")

    with bfio.BioReader(ff_path, max_workers=2) as bf:
        ff_image = bf[:, :, :, 0, 0].squeeze()

    if df_path is not None:
        with bfio.BioReader(df_path, max_workers=2) as df:
            df_image = df[:, :, :, 0, 0].squeeze()
    else:
        df_image = None

    batch_indices = list(range(0, len(img_paths), 16))
    if batch_indices[-1] != len(img_paths):
        batch_indices.append(len(img_paths))

    for i_start, i_end in tqdm.tqdm(
        zip(batch_indices[:-1], batch_indices[1:]),
        total=len(batch_indices) - 1,
    ):
        _unshade_batch(
            img_paths[i_start:i_end], out_dir, ff_image, df_image, keep_orig_dtype
        )


def _unshade_batch(
    batch_paths: typing.List[pathlib.Path],
    out_dir: pathlib.Path,
    ff_image: numpy.ndarray,
    df_image: typing.Optional[numpy.ndarray] = None,
    keep_orig_dtype: typing.Optional[bool] = True,
) -> None:
    """Apply flatfield correction to a batch of images.

    Args:
        batch_paths: list of paths to images to be processed
        out_dir: directory to save the corrected images
        ff_image: component to be used for flatfield correction
        df_image: component to be used for flatfield correction
        keep_orig_dtype: if True, the output images will be saved with the same
        dtype as the input images. If False, the output images will be saved as
        float32.
    """
    
    # Load images in parallel
    images = [None] * len(batch_paths)
    with ProcessPoolExecutor(max_workers=utils.MAX_WORKERS) as executor:
        load_futures = {
            executor.submit(utils.load_img, inp_path, i): i
            for i, inp_path in enumerate(batch_paths)
        }
        for future in as_completed(load_futures):
            idx, img = future.result()
            images[idx] = img

    img_stack = numpy.stack(images, axis=0).astype(numpy.float32)

    def get_min_max(img_stack):
        min_val = img_stack.min(axis=(-1, -2), keepdims=True)
        max_val = img_stack.max(axis=(-1, -2), keepdims=True)
        return min_val, max_val

    min_orig, max_orig = get_min_max(img_stack)

    if df_image is not None:
        img_stack -= df_image

    img_stack /= ff_image + 1e-8

    min_new, max_new = get_min_max(img_stack)
    img_stack = (img_stack - min_new) / (max_new - min_new) * (
        max_orig - min_orig
    ) + min_orig

    if keep_orig_dtype:
        orig_dtype = images[0].dtype
        if np.issubdtype(orig_dtype, np.integer):
            dtype_info = np.iinfo(orig_dtype)
            img_stack = np.clip(img_stack, dtype_info.min, dtype_info.max)
            img_stack = np.round(img_stack).astype(orig_dtype)
        elif np.issubdtype(orig_dtype, np.floating):
            img_stack = np.clip(img_stack, 0.0, 1.0)
            img_stack = img_stack.astype(orig_dtype)

    # Save images in parallel
    with ProcessPoolExecutor(max_workers=utils.MAX_WORKERS) as executor:
        save_futures = [
            executor.submit(utils.save_img, inp_path, img, out_dir)
            for inp_path, img in zip(batch_paths, img_stack)
        ]
        for future in as_completed(save_futures):
            future.result()  # raises any exceptions from worker