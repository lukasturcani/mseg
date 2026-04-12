from typing import Any

import numpy as np
import numpy.typing as npt
import plotly.express as px
import streamlit as st
from plotly import graph_objects as go
from scipy import stats

from mseg._internal.utils import parse_data_file


def main() -> None:
    st.title("Permutation ANOVA")
    data_file = st.file_uploader("Choose a file")
    if data_file is None:
        return
    data_df = parse_data_file(data_file.getvalue().decode()).sort("time")
    st.write(data_df)
    chart = px.line(data_df, x="time", y="power")
    st.plotly_chart(chart)

    with st.sidebar:
        st.header("Parameters")
        breakpoints_str = st.text_input(
            "Breakpoint times",
            help=(
                "Comma-separated list of time values at which to "
                "split the data into segments for comparison."
            ),
        )
        n_resamples = st.number_input(
            "n_resamples",
            value=9999,
            min_value=100,
            step=100,
            help="Number of permutation resamples.",
        )
        random_seed = st.number_input(
            "random_seed",
            value=0,
            step=1,
            help=(
                "Seed for the random number generator. Set to 0 for no seed."
            ),
        )

    if not breakpoints_str.strip():
        st.info("Enter breakpoint times in the sidebar to define segments.")
        return

    breakpoints = sorted(
        float(b.strip()) for b in breakpoints_str.split(",") if b.strip()
    )
    times = data_df["time"].to_numpy()
    power = data_df["power"].to_numpy()

    edges = [
        float(times[0]),
        *breakpoints,
        float(times[-1]) + 1,
    ]
    segments: list[npt.NDArray[np.float64]] = []
    segment_labels: list[str] = []
    for i in range(len(edges) - 1):
        mask = (times >= edges[i]) & (times < edges[i + 1])
        seg = power[mask]
        if len(seg) == 0:
            continue
        segments.append(seg)
        segment_labels.append(f"[{edges[i]:.2f}, {edges[i + 1]:.2f})")

    if len(segments) < 2:  # noqa: PLR2004
        st.error(
            "Need at least 2 non-empty segments. Adjust breakpoint times."
        )
        return

    st.header("Segments")
    for i, (label, seg) in enumerate(
        zip(segment_labels, segments, strict=True)
    ):
        st.write(
            f"**Segment {i + 1}** {label}: "
            f"n={len(seg)}, "
            f"mean={seg.mean():.4f}, "
            f"std={seg.std():.4f}"
        )

    seed = random_seed if random_seed != 0 else None
    result = _permutation_anova(
        tuple(segments),
        n_resamples=n_resamples,
        seed=seed,
    )
    st.header("Results")
    st.write(f"**F-statistic:** {result.statistic:.4f}")
    st.write(f"**p-value:** {result.pvalue:.6f}")

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, (label, seg_start, seg_end) in enumerate(
        zip(
            segment_labels,
            edges[:-1],
            edges[1:],
            strict=True,
        )
    ):
        mask = (times >= seg_start) & (times < seg_end)
        fig.add_trace(
            go.Scatter(
                x=times[mask],
                y=power[mask],
                mode="lines",
                name=f"Segment {i + 1} {label}",
                line={"color": colors[i % len(colors)]},
            )
        )
    for bp in breakpoints:
        fig.add_vline(
            x=bp,
            line_dash="dash",
            line_color="red",
        )
    fig.update_layout(
        title=(
            f"Permutation ANOVA: F={result.statistic:.4f}, "
            f"p={result.pvalue:.6f}"
        ),
        xaxis={"title": "time"},
        yaxis={"title": "power"},
    )
    st.header("Results Figure")
    st.plotly_chart(fig)

    st.header("Null Distribution")
    null_fig = go.Figure()
    null_fig.add_trace(
        go.Histogram(
            x=result.null_distribution,
            name="null distribution",
            nbinsx=50,
        )
    )
    null_fig.add_vline(
        x=result.statistic,
        line_dash="dash",
        line_color="red",
        annotation_text=f"observed F={result.statistic:.4f}",
    )
    null_fig.update_layout(
        xaxis={"title": "F-statistic"},
        yaxis={"title": "count"},
    )
    st.plotly_chart(null_fig)


def _f_statistic(
    *samples: npt.NDArray[np.float64],
) -> float:
    return float(stats.f_oneway(*samples).statistic)


@st.cache_resource
def _permutation_anova(
    segments: tuple[npt.NDArray[np.float64], ...],
    *,
    n_resamples: int,
    seed: int | None,
) -> Any:
    rng = np.random.default_rng(seed) if seed is not None else None
    return stats.permutation_test(
        segments,
        _f_statistic,
        n_resamples=n_resamples,
        alternative="greater",
        random_state=rng,
    )


if __name__ == "__main__":
    main()
