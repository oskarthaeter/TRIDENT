from abc import ABC, abstractmethod
import numpy as np


class BaseTile(ABC):
    """
    Abstract base class for Tiles.
    """

    @abstractmethod
    def importance(self) -> int:
        pass

    @abstractmethod
    def __lt__(self, other: "BaseTile") -> bool:
        pass

    @abstractmethod
    def __le__(self, other: "BaseTile") -> bool:
        pass

    @abstractmethod
    def __eq__(self, other: "BaseTile") -> bool:
        pass

    @abstractmethod
    def __ne__(self, other: "BaseTile") -> bool:
        pass

    @abstractmethod
    def __gt__(self, other: "BaseTile") -> bool:
        pass

    @abstractmethod
    def __ge__(self, other: "BaseTile") -> bool:
        pass


class Tile(BaseTile):
    """
    Represents a single tile with one offset and one bytecount.
    """

    def __init__(self, offset: int, bytecount: int, coord: np.ndarray, idx: int):
        self.offset = offset
        self.bytecount = bytecount
        self.coord = coord
        self.idx = idx

    def importance(self) -> int:
        return int(self.bytecount)

    def __lt__(self, other: "Tile") -> bool:
        return self.offset < other.offset

    def __le__(self, other: "Tile") -> bool:
        return self.offset <= other.offset

    def __eq__(self, other: "Tile") -> bool:
        return self.offset == other.offset

    def __ne__(self, other: "Tile") -> bool:
        return self.offset != other.offset

    def __gt__(self, other: "Tile") -> bool:
        return self.offset > other.offset

    def __ge__(self, other: "Tile") -> bool:
        return self.offset >= other.offset

    def __repr__(self):
        return f"Tile\n\toffsets: {self.offset}\n\tbytecounts: {self.bytecount}\n\tcoordinate: {self.coord}"


class AggregatedTile(BaseTile):
    """
    Represents an aggregated tile with n subtiles:
    - Holds n offsets and n bytecounts.
    - Aggregates bytecounts (sum) and offsets (minimum) when needed.
    """

    def __init__(self, offsets: list, bytecounts: list, coord: np.ndarray, idx: int):

        self.offset = offsets
        self.bytecount = bytecounts
        self.coord = coord
        self.idx = idx
        self.min_offset = min(offsets)
        self.importance_value = sum(bytecounts)

    def importance(self) -> int:
        return int(self.importance_value)

    def __lt__(self, other: "AggregatedTile") -> bool:
        return self.min_offset < other.min_offset

    def __le__(self, other: "AggregatedTile") -> bool:
        return self.min_offset <= other.min_offset

    def __eq__(self, other: "AggregatedTile") -> bool:
        return self.min_offset == other.min_offset

    def __ne__(self, other: "AggregatedTile") -> bool:
        return self.min_offset != other.min_offset

    def __gt__(self, other: "AggregatedTile") -> bool:
        return self.min_offset > other.min_offset

    def __ge__(self, other: "AggregatedTile") -> bool:
        return self.min_offset >= other.min_offset

    def __repr__(self):
        return f"AggregatedTile\n\toffsets: {self.offset}\n\tbytecounts: {self.bytecount}\n\tcoordinate: {self.coord}"
