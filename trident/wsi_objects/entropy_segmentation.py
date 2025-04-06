import numpy as np
import scipy as sp
import math
from pathlib import Path
from skimage.filters import rank, threshold_multiotsu
from skimage.morphology import remove_small_objects, disk
import PIL.Image
import matplotlib.pyplot as plt
from trident.wsi_objects.utils import (
    extract_page_infos,
)


def prep_heatmap(heatmap: np.ndarray):
    # Normalize heatmap to 0-1
    if heatmap.dtype != bool:
        heatmap = heatmap.astype(np.uint32)
        normed_heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        normed_heatmap = (normed_heatmap * 255).astype(np.uint8)

        # Apply a colormap and convert to RGB
        colormap = plt.get_cmap("viridis")
        colored_heatmap = (colormap(normed_heatmap)[:, :, :3] * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8) * 255
        colored_heatmap = np.stack([heatmap] * 3, axis=-1)

    return PIL.Image.fromarray(colored_heatmap)


def prep_segmentation(
    byte_counts: np.ndarray,
    tiles_x: int,
    tiles_y: int,
    lower_percentile: int = 5,
    upper_percentile: int = 99,
    disk_size: int = 128,
    s0: int = 100,
    s1: int = 100,
    verbose: bool = False,
):
    heatmap_list = []
    # Normalization
    lower_bound, upper_bound = np.percentile(
        byte_counts, [lower_percentile, upper_percentile]
    )
    clipped_byte_counts = np.clip(byte_counts, lower_bound, upper_bound)
    normed_byte_counts = (clipped_byte_counts - lower_bound) / (
        upper_bound - lower_bound
    )
    normed_byte_counts = (normed_byte_counts * 255).astype(np.uint8)
    byte_count_array = normed_byte_counts.reshape(tiles_y, tiles_x)

    # Save heatmap
    if verbose:
        heatmap_list.append(prep_heatmap(byte_count_array.copy()))

    # Apply bilateral filtering
    byte_count_array = rank.mean_bilateral(
        byte_count_array, disk(disk_size), s0=s0, s1=s1
    )

    if verbose:
        heatmap_list.append(prep_heatmap(byte_count_array.copy()))
        return byte_count_array, heatmap_list
    else:
        return byte_count_array


def segment_array_with_multiotsu(heatmap: np.ndarray):
    """
    Segments a tile byte count array into two regions using Otsu's thresholding.
    """

    # Apply Otsu's threshold directly
    otsu_threshold = threshold_multiotsu(heatmap, classes=5)[-2]
    segmented_array = heatmap >= otsu_threshold
    return segmented_array


def entropy_mask(
    filepath: Path,
    area_threshold=64,
    verbose: bool = False,
) -> dict:
    """
    Segments WSI into tissue foreground and background using entropy-based segmentation.

    Parameters:
        filepath (str): Path to the slide file.

    Returns:
        numpy.ndarray: Final combined segmentation mask.
    """

    # Extract tile information
    slide_infos = extract_page_infos(filepath, 0)
    tile_byte_counts = np.array(slide_infos["bytecounts"])
    image_width = slide_infos["width"]
    image_height = slide_infos["height"]
    tile_size = slide_infos["tile_size"]

    # Grid dimensions
    tiles_x = math.ceil(image_width / tile_size)
    tiles_y = math.ceil(image_height / tile_size)
    summed_tile_counts = tile_byte_counts

    heatmap = prep_segmentation(
        summed_tile_counts,
        tiles_x,
        tiles_y,
        lower_percentile=5,
        upper_percentile=99,
        verbose=verbose,
    )
    if verbose:
        heatmap, heatmap_list = heatmap

    # Perform segmentation
    mask = segment_array_with_multiotsu(heatmap)

    # mask = remove_small_objects(
    #     mask, min_size=area_threshold, connectivity=2
    # )

    if verbose:
        return mask, heatmap_list
    else:
        return mask
