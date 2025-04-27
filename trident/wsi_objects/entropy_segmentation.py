import numpy as np
import math
from pathlib import Path
from skimage.filters import rank, threshold_multiotsu, threshold_otsu
from skimage.color import rgb2hsv
from skimage.morphology import disk, opening, closing
import PIL.Image
import openslide
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
):
    lower_percentile = 5
    upper_percentile = 99
    disk_size = 128
    s0 = 100
    s1 = 100

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

    # Apply bilateral filtering
    byte_count_array = rank.mean_bilateral(
        byte_count_array, disk(disk_size), s0=s0, s1=s1
    )

    return byte_count_array


def segment_array_with_otsu(heatmap: np.ndarray):
    """
    Segments a tile byte count array into two regions using Otsu's thresholding.
    """

    # Apply Otsu's threshold directly
    otsu_threshold = threshold_otsu(heatmap)
    segmented_array = heatmap >= otsu_threshold
    return segmented_array


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
) -> np.ndarray:
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
    )

    # Perform segmentation
    mask = segment_array_with_otsu(heatmap)

    # opening to remove small objects
    selem = disk(1)
    mask = opening(mask.astype(np.uint8), footprint=selem)
    mask = mask.astype(np.uint8)
    mask = np.clip(mask, 0, 1)

    return mask


def thumbnail_segmentation(
    slide_path: str,
    get_thumbnail_fn: callable = None,
):
    mask = entropy_mask(slide_path)
    height, width = mask.shape

    # Load the WSI
    if get_thumbnail_fn is not None:
        thumbnail = get_thumbnail_fn((height, width))
    else:
        with openslide.OpenSlide(slide_path) as slide:
            thumbnail = slide.get_thumbnail((width, height))

    # Ensure that the thumbnail is actually the correct size
    if thumbnail.size != (width, height):
        thumbnail = thumbnail.resize((width, height))

    masked_thumbnail = np.array(thumbnail)
    # Set everything in the thumbnail to white where mask is 0
    masked_thumbnail[mask == 0] = [255, 255, 255]

    # Convert to HSV
    masked_hue_channel = rgb2hsv(masked_thumbnail)[:, :, 0]

    # Threshold
    hue_threshold = threshold_otsu(masked_hue_channel)
    final_mask = masked_hue_channel > hue_threshold

    return final_mask
