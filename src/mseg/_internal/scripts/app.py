import numpy as np
import polars as pl
import Rbeast as rb  # noqa: N813
import streamlit as st

from mseg._internal.utils import parse_data_file


def main() -> None:
    st.title("Segmentation Analysis")
    data_file = st.file_uploader("Choose a file")
    if data_file is None:
        return
    data_df = parse_data_file(data_file.getvalue().decode()).sort("time")
    delta_t = _get_delta_t(data_df)
    if delta_t is None:
        st.error("Time delta between samples is not constant.")
        return
    else:
        st.success(f"Time delta between samples is {delta_t:.2f} seconds.")
    st.write(data_df)
    st.line_chart(data_df, x="time", y="power")

    with st.sidebar:
        st.header("RBeast Parameters")
        has_outlier = st.checkbox("Has Outlier", value=False)
        mcmc_samples = st.number_input(
            "MCMC Samples", value=80000, min_value=1
        )
        mcmc_chains = st.number_input("MCMC Chains", value=5, min_value=1)
        seed = st.number_input("Random Seed", value=32, min_value=0)
        rb.beast(
            Y=data_df["power"].to_numpy(),
            start=st.number_input(
                "Start",
                value=data_df.select(pl.col("time").min()).item(),
                min_value=0.0,
            ),
            deltat=st.number_input(
                "DeltaT",
                value=delta_t,
            ),
            season=st.selectbox(
                "Season",
                [
                    "harmonic",
                    "dummy",
                    "svd",
                    "none",
                ],
                index=3,
            ),
        )


def _get_delta_t(data: pl.DataFrame, tolerance: float = 1e-3) -> float | None:
    diffs = data["time"].diff().drop_nulls()
    ref = diffs[0]
    all_close = np.all(np.isclose(diffs.to_numpy(), ref, atol=tolerance))
    if all_close:
        return ref
    return None


if __name__ == "__main__":
    main()
