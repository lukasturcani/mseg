"""Ruptures tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ruptures as rpt
from plotly import graph_objects as go

if TYPE_CHECKING:
    import polars as pl


def ruptures(data: pl.DataFrame, *, pen: int = 10) -> list[float]:
    """Detect change points using ruptures."""
    signal = data["power"].to_numpy()
    algo = rpt.Pelt(model="l2").fit(signal)
    breakpoints = algo.predict(pen=pen)
    change_points = data["time"][breakpoints[:-1]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=data["power"],
            name="Time Series",
            line={"color": "blue"},
        )
    )
    for cp in change_points:
        fig.add_vline(
            x=cp,
            line_color="red",
            name="Detected Change Point",
        )
    fig.update_layout(
        title="Automatic Change Point Detection using ruptures (Pelt)",
        xaxis_title="Time",
        yaxis_title="Power",
        showlegend=True,
        width=1200,
        height=600,
    )
    fig.show()
    return list(change_points)
