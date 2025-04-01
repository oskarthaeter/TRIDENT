import tifffile
from pathlib import Path


def extract_bytecounts(tif_path: Path) -> tuple[int, ...]:
    with tifffile.TiffFile(tif_path) as tif:
        return tif.pages[0].tags["TileByteCounts"].value


def extract_tile_offsets(tif_path: Path) -> tuple[int, ...]:
    with tifffile.TiffFile(tif_path) as tif:
        return tif.pages[0].tags["TileOffsets"].value


def extract_JPEGTables(tif_path: Path) -> bytes | None:
    with tifffile.TiffFile(tif_path) as tif:
        if "JPEGTables" not in tif.pages[0].tags:
            return None
        else:
            return tif.pages[0].tags["JPEGTables"].value


def extract_tile_size(tif_path: Path) -> int:
    with tifffile.TiffFile(tif_path) as tif:
        tile_width = tif.pages[0].tags["TileWidth"].value
        tile_height = tif.pages[0].tags["TileLength"].value
        assert tile_width == tile_height
        return tile_width


def extract_root_image_size(tif_path: Path) -> tuple[int, int]:
    with tifffile.TiffFile(tif_path) as tif:
        return (
            tif.pages[0].tags["ImageWidth"].value,
            tif.pages[0].tags["ImageLength"].value,
        )


def extract_page_infos(tif_path: Path, level: int) -> dict:
    # only count pages that are tiled
    page_counter = 0
    with tifffile.TiffFile(tif_path) as tif:
        for page in tif.pages:
            if "TileWidth" not in page.tags:
                continue
            if page_counter == level:
                bytecounts = page.tags["TileByteCounts"].value
                offsets = page.tags["TileOffsets"].value
                tile_size = page.tags["TileWidth"].value
                image_width, image_height = (
                    page.tags["ImageWidth"].value,
                    page.tags["ImageLength"].value,
                )
                return {
                    "bytecounts": bytecounts,
                    "offsets": offsets,
                    "tile_size": tile_size,
                    "width": image_width,
                    "height": image_height,
                }
            elif page_counter > level:
                break
            else:
                page_counter += 1
    raise ValueError(f"Page {level} not found")
