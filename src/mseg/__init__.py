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


def pyearth() -> None:
    # Create some fake data
    np.random.seed(0)
    m = 1000
    n = 10
    X = 80 * np.random.uniform(size=(m, n)) - 40
    y = np.abs(X[:, 6] - 4.0) + 1 * np.random.normal(size=m)

    # Fit an Earth model
    model = Earth()
    model.fit(X, y)

    # Print the model
    print(model.trace())
    print(model.summary())

    # Plot the model
    y_hat = model.predict(X)
    pyplot.figure()
    pyplot.plot(X[:, 6], y, "r.")
    pyplot.plot(X[:, 6], y_hat, "b.")
    pyplot.xlabel("x_6")
    pyplot.ylabel("y")
    pyplot.title("Simple Earth Example")
    pyplot.show()


__all__ = ["read_data"]
