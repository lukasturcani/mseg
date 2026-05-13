from itertools import combinations
from typing import Any

import numpy as np
import numpy.typing as npt
import plotly.express as px
import polars as pl
import streamlit as st
from plotly import graph_objects as go
from scipy import stats

from mseg._internal.utils import parse_tabular_data_file


def main() -> None:
    st.title("Repeated-Measures Permutation ANOVA")
    data_file = st.file_uploader("Choose a file")
    if data_file is None:
        return
    data_df = parse_tabular_data_file(data_file.getvalue().decode())
    if data_df.is_empty():
        st.error("File contains no data.")
        return

    subject_col = data_df.columns[0]
    condition_cols = data_df.columns[1:]

    st.header("Data")
    st.write(data_df)

    if len(condition_cols) < 2:  # noqa: PLR2004
        st.error("Need at least 2 condition columns.")
        return

    with st.sidebar:
        st.header("Parameters")
        selected_conditions = st.multiselect(
            "Conditions to compare",
            options=condition_cols,
            default=list(condition_cols),
            help="Pick at least 2 condition columns to include in the test.",
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

    if len(selected_conditions) < 2:  # noqa: PLR2004
        st.info("Select at least 2 conditions in the sidebar.")
        return

    samples = tuple(
        data_df[col].to_numpy().astype(np.float64)
        for col in selected_conditions
    )
    subjects = data_df[subject_col].to_list()

    st.header("Conditions")
    for label, values in zip(selected_conditions, samples, strict=True):
        st.write(
            f"**{label}**: n={len(values)}, "
            f"mean={values.mean():.4f}, std={values.std(ddof=1):.4f}"
        )

    st.header("Per-Subject Trajectories")
    traj_fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, subject in enumerate(subjects):
        traj_fig.add_trace(
            go.Scatter(
                x=list(selected_conditions),
                y=[float(values[i]) for values in samples],
                mode="lines+markers",
                name=str(subject),
                line={"color": colors[i % len(colors)]},
            )
        )
    traj_fig.update_layout(
        xaxis={"title": "condition"},
        yaxis={"title": "value"},
    )
    st.plotly_chart(traj_fig)

    seed = random_seed if random_seed != 0 else None
    result = _rm_permutation_anova(
        samples,
        n_resamples=n_resamples,
        seed=seed,
    )

    st.header("Results")
    st.write(f"**F-statistic:** {result.statistic:.4f}")
    st.write(f"**p-value:** {result.pvalue:.6f}")

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

    st.header("Pairwise Comparisons")
    st.caption(
        "Paired permutation tests on the differences within each rat, "
        "with Holm-Bonferroni correction across all pairs."
    )
    pairs = list(combinations(range(len(selected_conditions)), 2))
    raw_pvalues: list[float] = []
    mean_diffs: list[float] = []
    pair_labels: list[str] = []
    for i, j in pairs:
        a = samples[i]
        b = samples[j]
        pair_result = _paired_permutation_test(
            a,
            b,
            n_resamples=n_resamples,
            seed=seed,
        )
        raw_pvalues.append(float(pair_result.pvalue))
        mean_diffs.append(float(np.mean(a - b)))
        pair_labels.append(
            f"{selected_conditions[i]} vs {selected_conditions[j]}"
        )
    adjusted = _holm_bonferroni(raw_pvalues)
    pairwise_df = pl.DataFrame(
        {
            "comparison": pair_labels,
            "mean_difference": mean_diffs,
            "p_raw": raw_pvalues,
            "p_holm": adjusted,
            "significant_holm_0.05": [p < 0.05 for p in adjusted],  # noqa: PLR2004
        }
    )
    st.write(pairwise_df)


def _rm_anova_f(*samples: npt.NDArray[np.float64]) -> float:
    data = np.stack(samples, axis=1)
    n_subjects, n_conditions = data.shape
    grand_mean = data.mean()
    condition_means = data.mean(axis=0)
    subject_means = data.mean(axis=1)

    ss_condition = n_subjects * np.sum(
        (condition_means - grand_mean) ** 2
    )
    ss_subject = n_conditions * np.sum(
        (subject_means - grand_mean) ** 2
    )
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = ss_total - ss_condition - ss_subject

    df_condition = n_conditions - 1
    df_error = (n_subjects - 1) * (n_conditions - 1)

    ms_condition = ss_condition / df_condition
    ms_error = ss_error / df_error
    return float(ms_condition / ms_error)


@st.cache_data
def _rm_permutation_anova(
    samples: tuple[npt.NDArray[np.float64], ...],
    *,
    n_resamples: int,
    seed: int | None,
) -> Any:
    rng = np.random.default_rng(seed) if seed is not None else None
    return stats.permutation_test(
        samples,
        _rm_anova_f,
        n_resamples=n_resamples,
        permutation_type="samples",
        alternative="greater",
        random_state=rng,
    )


def _mean_difference(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> float:
    return float(np.mean(a - b))


@st.cache_data
def _paired_permutation_test(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    *,
    n_resamples: int,
    seed: int | None,
) -> Any:
    rng = np.random.default_rng(seed) if seed is not None else None
    return stats.permutation_test(
        (a, b),
        _mean_difference,
        n_resamples=n_resamples,
        permutation_type="samples",
        alternative="two-sided",
        random_state=rng,
    )


def _holm_bonferroni(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        scaled = (m - rank) * pvalues[idx]
        running_max = max(running_max, scaled)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


if __name__ == "__main__":
    main()
