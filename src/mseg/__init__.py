"""Segmentation analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import polars as pl
from matplotlib import pyplot as plt
from pyearth import Earth

if TYPE_CHECKING:
    from pathlib import Path


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


def pyearth(data: pl.DataFrame) -> Earth:
    x = data["time"].to_numpy().reshape(-1, 1)
    y = data["power"].to_numpy()

    # Fit an Earth model
    model = Earth()
    model.fit(x, y)

    # Print the model
    print(model.trace())  # noqa: T201
    print(model.summary())  # noqa: T201

    # Plot the model
    y_hat = model.predict(x)
    plt.figure()
    plt.plot(x, y, "r.")
    plt.plot(x, y_hat, "b.")
    plt.xlabel("time")
    plt.ylabel("power")
    plt.title("Simple Earth Example")
    plt.show()
    return model


def knots(model: Earth) -> list[float]:
    basis_functions = model.basis_
    return [bf.get_knot() for bf in basis_functions]


__all__ = ["read_data"]
