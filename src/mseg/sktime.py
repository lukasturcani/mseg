"""sktime tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from plotly import graph_objects as go
from sktime.annotation.ggs import GreedyGaussianSegmentation

if TYPE_CHECKING:
    import polars as pl


def eagglo(
    data: pl.DataFrame,
    alpha: float = 1.0,
    penalty: Literal["mean_diff_penalty", "len_penalty"] | None = None,
) -> list[float]:
    """Detect change points using EAgglo."""
    signal = data["power"].to_numpy()
    algo = EAgglo(alpha=alpha, penalty=penalty)
    arr = algo.fit_transform(signal)
    indices = np.where(np.diff(arr) != 0)[0] + 1
    change_points = data["time"][indices]

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
        title="Automatic Change Point Detection using EAgglo",
        xaxis_title="Time",
        yaxis_title="Power",
        showlegend=True,
        width=1200,
        height=600,
    )
    fig.show()
    return list(change_points)


def greedy_gaussian_segmentation(
    data: pl.DataFrame,
    k_max: int = 5,
) -> list[float]:
    """Detect change points using GreedyGaussianSegmentation."""
    signal = data["power"].to_numpy()
    algo = GreedyGaussianSegmentation(k_max=k_max)
    arr = algo.fit_predict(np.array([[y] for y in signal]))
    indices = np.where(np.diff(arr) != 0)[0] + 1
    change_points = data["time"][indices]

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
        title="Automatic Change Point Detection using GreedyGaussianSegmentation",
        xaxis_title="Time",
        yaxis_title="Power",
        showlegend=True,
        width=1200,
        height=600,
    )
    fig.show()
    return list(change_points)
