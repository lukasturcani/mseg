"""Segmentation analysis."""

from mseg import rbeast, ruptures
from mseg._internal.utils import line_plot, read_data, scatter_plot

__all__ = [
    "line_plot",
    "rbeast",
    "read_data",
    "ruptures",
    "scatter_plot",
]
