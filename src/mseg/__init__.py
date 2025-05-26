"""Segmentation analysis."""

from mseg import rbeast, ruptures
from mseg._internal.utils import line_plot, parse_data_file, scatter_plot

__all__ = [
    "line_plot",
    "parse_data_file",
    "rbeast",
    "ruptures",
    "scatter_plot",
]
