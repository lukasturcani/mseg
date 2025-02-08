"""Segmentation analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import polars as pl
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from pathlib import Path

    from pyearth import Earth


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
    return pl.DataFrame({"time": xs, "power": ys})


def scatter_plot(data: pl.DataFrame) -> None:
    """Plot data."""
    fig = px.scatter(data, x="time", y="power")
    fig.show()


def line_plot(data: pl.DataFrame) -> None:
    """Plot data."""
    fig = px.line(data, x="time", y="power")
    fig.show()


def pyearth(model: Earth, data: pl.DataFrame) -> Earth:
    x = data["time"].to_numpy().reshape(-1, 1)
    y = data["power"].to_numpy()

    # Fit an Earth model
    model.fit(x, y)

    # Print the model
    print(model.trace())  # noqa: T201
    print(model.summary())  # noqa: T201

    # Plot the model
    y_hat = model.predict(x)
    fig = px.scatter()
    fig.add_scatter(
        x=x.flatten(),
        y=y,
        mode="markers",
        name="Actual",
        marker={"color": "red"},
    )
    fig.add_scatter(
        x=x.flatten(),
        y=y_hat,
        mode="markers",
        name="Predicted",
        marker={"color": "blue"},
    )
    fig.update_layout(
        title="PyEarth Example", xaxis_title="time", yaxis_title="power"
    )
    fig.show()
    return model


def knots(model: Earth) -> list[float]:
    return [
        bf.get_knot()
        for bf in model.basis_
        if hasattr(bf, "get_knot") and not bf.is_pruned()
    ]


__all__ = ["read_data"]
