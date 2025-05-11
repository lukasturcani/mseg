"""Rbeast tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import Rbeast as rb  # noqa: N813
from plotly import graph_objects as go

if TYPE_CHECKING:
    import polars as pl


def rbeast(data: pl.DataFrame) -> list[float]:
    """Detect change points using ruptures."""
    signal = data["power"].to_numpy()
    output = rb.beast(
        signal,
        season="none",
        hasOutlier=True,
        print_param=False,
    )
    breakpoints = [
        _ChangePoint(
            location=data["time"][int(location)],
            probability=probability,
        )
        for location, probability in zip(output.trend.cp, output.trend.cpPr)
    ]
    breakpoints.sort(key=lambda x: x.location)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=data["power"],
            name="Time Series",
            line={"color": "blue"},
        )
    )
    for bp in breakpoints:
        fig.add_vline(
            x=bp.location,
            line_color="red",
            name="Detected Change Point",
        )
    fig.update_layout(
        title="Rbeast",
        xaxis_title="Time",
        yaxis_title="Power",
        showlegend=True,
        width=1200,
        height=600,
    )
    fig.show()
    return list(breakpoints)


@dataclass
class _ChangePoint:
    location: int
    probability: float
