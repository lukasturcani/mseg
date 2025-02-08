"""Segmentation analysis."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Data:
    """A dataset."""

    xs: list[float]
    """The x coordinates of the data points."""
    ys: list[float]
    """The y coordinates of the data points."""


def load_data(path: Path) -> Data:
    """Load data from a file."""
    content = path.read_text()
    lines = content.splitlines()
    xs = []
    ys = []
    for line in lines:
        x, y = line.split()
        xs.append(float(x))
        ys.append(float(y))
    return Data(xs, ys)


__all__ = [
    "Data",
    "load_data",
]
