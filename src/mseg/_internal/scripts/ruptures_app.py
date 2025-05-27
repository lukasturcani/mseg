from collections.abc import Sequence

import plotly.express as px
import polars as pl
import ruptures as rpt
import streamlit as st
from plotly import graph_objects as go

from mseg._internal.utils import parse_data_file


def main() -> None:
    st.title("Ruptures Breakpoint Detection")
    data_file = st.file_uploader("Choose a file")
    if data_file is None:
        return
    data_df = parse_data_file(data_file.getvalue().decode()).sort("time")
    st.write(data_df)
    chart = px.line(data_df, x="time", y="power")
    st.plotly_chart(chart)

    with st.sidebar:
        st.header("Parameters")
        breakpoint_idxs = rpt.Pelt(
            model=st.selectbox(
                "model",
                ["l1", "l2", "rbf"],
                index=1,
                help="segment model",
            ),
            min_size=st.number_input(
                "min_size",
                value=2,
                help="minimum segment length",
            ),
            jump=st.number_input(
                "jump",
                value=1,
                help="subsamble (one every *jump* points)",
            ),
        ).fit_predict(
            signal=data_df["power"].to_numpy(),
            pen=st.number_input(
                "pen",
                value=200,
                help="penalty value",
            ),
        )
    breakpoints = data_df["time"][breakpoint_idxs[:-1]]
    st.header("Breakpoints")
    st.dataframe(
        pl.DataFrame(
            {
                "": list(range(1, len(breakpoints) + 1)),
                "time": breakpoints,
            }
        )
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data_df["time"].to_numpy(),
            y=data_df["power"].to_numpy(),
            mode="lines",
            name="power",
            yaxis="y",
        )
    )
    _draw_breakpoints(fig, breakpoints, "breakpoints")
    fig.update_layout(
        xaxis={
            "title": "time",
        },
        yaxis={
            "title": "power",
        },
        yaxis2={
            "overlaying": "y",
            "side": "right",
        },
    )
    st.header("Results Figure")
    st.plotly_chart(fig)


def _draw_breakpoints(
    fig: go.Figure,
    breakpoints: Sequence[float],
    name: str,
) -> None:
    x_vals = []
    y_vals = []
    for time in breakpoints:
        x_vals.extend([time, time, None])
        y_vals.extend([0, 1, None])
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name=name,
            line={
                "color": "red",
                "dash": "dash",
            },
            yaxis="y2",
        )
    )


if __name__ == "__main__":
    main()
