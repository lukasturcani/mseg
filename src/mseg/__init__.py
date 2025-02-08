"""Segmentation analysis."""

from pathlib import Path

import plotly.express as px
import numpy as np
from pyearth import Earth
from matplotlib import pyplot
import polars as pl


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


def pyearth(data: pl.DataFrame) -> None:
    X = data["time"].to_numpy().reshape(-1, 1)
    y = data["power"].to_numpy()

    # Fit an Earth model
    model = Earth()
    model.fit(X, y)

    # Print the model
    print(model.trace())
    print(model.summary())

    # Plot the model
    y_hat = model.predict(X)
    pyplot.figure()
    pyplot.plot(X, y, "r.")
    pyplot.plot(X, y_hat, "b.")
    pyplot.xlabel("time")
    pyplot.ylabel("power")
    pyplot.title("Simple Earth Example")
    pyplot.show()


__all__ = ["read_data"]
