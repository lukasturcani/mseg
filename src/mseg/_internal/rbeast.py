from dataclasses import dataclass

import polars as pl
import Rbeast as rb  # noqa: N813
from plotly import graph_objects as go


def rbeast(
    data: pl.DataFrame,
    *,
    has_outlier: bool,
    mcmc_samples: int = 80000,
    mcmc_chains: int = 5,
    seed: int = 32,
) -> list[float]:
    """Detect change points using ruptures."""
    signal = data["power"].to_numpy()
    output = rb.beast(
        signal,
        season="none",
        hasOutlier=has_outlier,
        print_param=False,
        mcmc_seed=seed,
        mcmc_samples=mcmc_samples,
        mcmc_chains=mcmc_chains,
        ocp_minmax=(0, 20),
        scp_minmax=(0, 20),
        tcp_minmax=(0, 20),
    )
    breakpoints = [
        _ChangePoint(
            location=data["time"][int(location)],
            probability=probability,
        )
        for location, probability in zip(
            output.trend.cp, output.trend.cpPr, strict=True
        )
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
