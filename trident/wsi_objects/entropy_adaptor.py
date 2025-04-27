import numpy as np
import math
from pathlib import Path
import h5py
from trident.wsi_objects.tiles import Tile, AggregatedTile
from trident.wsi_objects.utils import (
    extract_page_infos,
    extract_JPEGTables,
    extract_root_image_size,
)


def coord_to_tileindex(
    coord: np.ndarray, tile_size: int, slide_width: int, slide_height: int
) -> int:
    """
    Convert a coordinate to a tile index.
    """
    tiles_x = math.ceil(slide_width / tile_size)
    x, y = coord
    x = int(x / tile_size)
    y = int(y / tile_size)
    return y * tiles_x + x


def coords_to_tiles(
    slide_path: Path, coords: np.ndarray, patch_size: int, level: int
) -> list:
    """
    Convert pixel slide coordinates to tiles.
    """
    slide_infos = extract_page_infos(slide_path, level)
    slide_width, slide_height = slide_infos["width"], slide_infos["height"]
    tile_size = slide_infos["tile_size"]
    tiles_x = math.ceil(slide_width / tile_size)

    offsets = slide_infos["offsets"]
    bytecounts = slide_infos["bytecounts"]

    tiles = []
    if patch_size == tile_size:
        # Create normal tile for each coordinate
        for coord in coords:
            tile_index = coord_to_tileindex(coord, tile_size, slide_width, slide_height)
            tile_offset = offsets[tile_index]
            tile_bytecount = bytecounts[tile_index]
            tiles.append(Tile(tile_offset, tile_bytecount, coord, tile_index))
    elif patch_size % tile_size == 0:
        # Create an aggregate tile for each coordinate
        factor = patch_size // tile_size
        for coord in coords:
            tile_index = coord_to_tileindex(coord, tile_size, slide_width, slide_height)
            subtile_offsets = []
            subtile_bytecounts = []
            for i in range(factor):
                for j in range(factor):
                    subtile_index = tile_index + i * tiles_x + j
                    subtile_offsets.append(offsets[subtile_index])
                    subtile_bytecounts.append(bytecounts[subtile_index])
            tiles.append(
                AggregatedTile(subtile_offsets, subtile_bytecounts, coord, tile_index)
            )
    else:
        raise ValueError("invalid patch_size")
    return tiles


def save_entropy_patches(
    slide_path: Path,
    patches_path: Path,
    assets: dict,
    attributes: dict,
    level: int,
):
    """
    Convert patch coords to format readable by tile-aligned dataloader.

    Args:
        slide_path (Path): Path to the whole-slide image file.
        clam_patches_path (Path): Path to the HDF5 file containing CLAM patches.
        output_dir (Path): Directory to save the output HDF5 file.
        assets (dict): Dictionary containing the original coords.
        attributes (dict): Dictionary containing the original attributes.
        level (int): Extraction level used by the patcher.
    """

    # load CLAM coordinates

    coords = assets["coords"]
    patch_size = attributes["patch_size"]

    # Extract slide info
    slide_infos = extract_page_infos(slide_path, level)
    slide_width, slide_height = slide_infos["width"], slide_infos["height"]
    original_tile_size = slide_infos["tile_size"]
    if patch_size < original_tile_size:
        raise ValueError(
            f"Patch size {patch_size} is smaller than original tile size {original_tile_size}."
        )
    if patch_size % original_tile_size != 0:
        # make the patch_size a multiple of the original tile size
        patch_size = math.ceil(patch_size / original_tile_size) * original_tile_size
        print(
            f"Patch size is not a multiple of the native tile size. Adjusted patch size to {patch_size}."
        )
        attributes["patch_size"] = patch_size
        if (
            "level0_magnification" in attributes
            and "target_magnification" in attributes
        ):
            attributes["patch_size_level0"] = (
                patch_size
                * attributes["level0_magnification"]
                // attributes["target_magnification"]
            )

    if level != 0:
        # convert root level coordinates to the specified level
        root_width, root_height = extract_root_image_size(slide_path)
        width_ratio = slide_width / root_width
        height_ratio = slide_height / root_height
        coords = np.floor(coords * np.array([width_ratio, height_ratio])).astype(
            np.int32
        )
        # we will have to revert the coordinates to the root level later
        revert_coords = lambda coords: np.floor(
            coords / np.array([width_ratio, height_ratio])
        ).astype(np.int32)
    elif np.any(coords[:, 0] >= slide_width) or np.any(coords[:, 1] >= slide_height):
        # check whether the coords are within the slide boundaries
        raise ValueError(
            "Coordinates exceed slide boundaries. Please check the slide level."
        )
    else:
        revert_coords = lambda coords: coords

    # Ensure coords are tile-aligned (multiple of original_tile_size)
    backward_aligned_coords = (coords // original_tile_size) * original_tile_size
    forward_aligned_coords = (
        np.ceil(coords / original_tile_size).astype(int) * original_tile_size
    )

    # Choose the closer alignment (backward or forward)
    dist_backward = np.abs(coords - backward_aligned_coords)
    dist_forward = np.abs(coords - forward_aligned_coords)

    # Use backward alignment if it's closer, otherwise use forward alignment
    aligned_coords = np.where(
        dist_backward <= dist_forward, backward_aligned_coords, forward_aligned_coords
    )

    # Filter out coordinates that exceed the valid slide boundaries
    valid_coords = aligned_coords[
        (aligned_coords[:, 0] <= slide_width - patch_size)
        & (aligned_coords[:, 1] <= slide_height - patch_size)
    ]

    # Convert aligned coordinates to tiles
    tiles = coords_to_tiles(slide_path, valid_coords, patch_size, level)

    # Extract JPEG tables
    jpeg_tables = extract_JPEGTables(slide_path)

    # Save results to an HDF5 file
    with h5py.File(patches_path, "w") as f:
        num_subtiles = int((patch_size // original_tile_size) ** 2)
        # Prepare datasets
        offsets = np.array([tile.offset for tile in tiles], dtype=np.uint64).reshape(
            -1, num_subtiles
        )
        f.create_dataset("offsets", data=offsets, dtype="uint64")

        bytecounts = np.array(
            [tile.bytecount for tile in tiles], dtype=np.uint32
        ).reshape(-1, num_subtiles)
        f.create_dataset("bytecounts", data=bytecounts, dtype="uint32")

        coords = revert_coords(
            np.array([tile.coord for tile in tiles], dtype=np.uint32).reshape(-1, 2)
        )
        coord_dataset = f.create_dataset("coords", data=coords, dtype="uint32")
        coord_dataset.attrs.update(attributes)

        # Add JPEG tables if present
        if jpeg_tables is not None:
            f.create_dataset(
                "jpeg_tables",
                data=np.frombuffer(jpeg_tables, dtype=np.uint8),
                dtype="uint8",
            )

        # Store metadata as attributes
        f.attrs["original_tile_size"] = original_tile_size
        f.attrs["target_tile_size"] = patch_size
        f.attrs["width"] = slide_width
        f.attrs["height"] = slide_height
        f.attrs["num_tiles"] = len(tiles)
        f.attrs["aggregated"] = original_tile_size != patch_size
        f.attrs["use_jpeg_tables"] = jpeg_tables is not None
