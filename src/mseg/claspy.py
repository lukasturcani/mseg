"""Claspy tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from claspy.segmentation import BinaryClaSPSegmentation
from plotly import graph_objects as go

if TYPE_CHECKING:
    import polars as pl


def claspy(data: pl.DataFrame) -> list[float]:
    """Detect change points using Clasp."""
    clasp = BinaryClaSPSegmentation(early_stopping=False)
    change_points = data["time"][clasp.fit_predict(data["power"].to_numpy())]

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
        title="Automatic Change Point Detection using ClasPy",
        xaxis_title="Time",
        yaxis_title="Power",
        showlegend=True,
        width=1200,
        height=600,
    )
    fig.show()
    return list(change_points)
