import numpy as np
import math
from pathlib import Path
from trident.wsi_objects.tiles import Tile, AggregatedTile
from trident.wsi_objects.utils import extract_page_infos


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
    slide_path: Path,
    coords: np.ndarray,
    patch_size: int,
) -> list:
    """
    Convert pixel slide coordinates to tiles.
    """
    slide_infos = extract_page_infos(slide_path, 0)
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
