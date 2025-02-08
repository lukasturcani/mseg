"""Segmentation analysis."""

from pathlib import Path

import polars as pl
import plotly.express as px


def read_data(path: Path) -> pl.DataFrame:
    """Read data from a path."""
    content = path.read_text()
    lines = content.splitlines()
    xs = []
    ys = []
    for line in lines:
        x, y = line.split()
        xs.append(float(x))
        ys.append(float(y))
    return pl.DataFrame({"x": xs, "y": ys})


def scatter_plot(data: pl.DataFrame) -> None:
    """Plot data."""
    fig = px.scatter(data, x="x", y="y")
    fig.show()


def line_plot(data: pl.DataFrame) -> None:
    """Plot data."""
    fig = px.line(data, x="x", y="y")
    fig.show()


__all__ = ["read_data"]
