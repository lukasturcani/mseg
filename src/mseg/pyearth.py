"""PyEarth tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px

if TYPE_CHECKING:
    import polars as pl
    from pyearth import Earth


def pyearth(model: Earth, data: pl.DataFrame) -> Earth:
    """Fit an Earth model."""
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
    """Get the knots (breakpoints) of the model."""
    return sorted(
        bf.get_knot()
        for bf in model.basis_
        if hasattr(bf, "get_knot") and not bf.is_pruned()
    )
