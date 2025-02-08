"""Ruptures tools."""

import polars as pl
import ruptures as rpt
from plotly import graph_objects as go


def ruptures(data: pl.DataFrame) -> None:
    """Detect change points using ruptures."""
    signal = data["power"].to_numpy()
    algo = rpt.Pelt(model="l2").fit(signal)
    breakpoints = algo.predict(pen=10)
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
            line_dash="dash",
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
