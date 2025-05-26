import numpy as np
import polars as pl
import Rbeast as rb  # noqa: N813
import streamlit as st

from mseg._internal.utils import parse_data_file


def main() -> None:
    st.title("RBeast Change Point Detection")
    data_file = st.file_uploader("Choose a file")
    if data_file is None:
        return
    data_df = parse_data_file(data_file.getvalue().decode()).sort("time")
    delta_t = _get_delta_t(data_df)
    if delta_t is None:
        st.error("Time delta between samples is not constant.")
        return
    st.success(f"Calculated DeltaT between samples is {delta_t:.2f}")
    st.write(data_df)
    st.line_chart(data_df, x="time", y="power")

    with st.sidebar:
        st.header("Parameters")
        rb.beast(
            Y=data_df["power"].to_numpy(),
            start=st.number_input(
                "start",
                value=data_df.select(pl.col("time").min()).item(),
                min_value=0.0,
            ),
            deltat=st.number_input(
                "deltat",
                value=delta_t,
            ),
            season=st.selectbox(
                "season",
                [
                    "harmonic",
                    "dummy",
                    "svd",
                    "none",
                ],
                index=3,
            ),
            period=st.number_input(
                "period",
                value=float("nan"),
            ),
            scp_minmax=st.slider(
                "scp_minmax",
                min_value=0,
                max_value=50,
                step=1,
                value=(0, 10),
            ),
            sorder_minmax=st.slider(
                "sorder_minmax",
                min_value=0,
                max_value=10,
                step=1,
                value=(0, 5),
            ),
            sseg_minlength=st.number_input(
                "sseg_minlength",
                value=None,
            ),
            sseg_leftmargin=st.number_input(
                "sseg_leftmargin",
                value=None,
            ),
            sseg_rightmargin=st.number_input(
                "sseg_rightmargin",
                value=None,
            ),
            tcp_minmax=st.slider(
                "tcp_minmax",
                min_value=0,
                max_value=50,
                step=1,
                value=(0, 20),
            ),
            torder_minmax=st.slider(
                "torder_minmax",
                min_value=0,
                max_value=10,
                step=1,
                value=(0, 1),
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
