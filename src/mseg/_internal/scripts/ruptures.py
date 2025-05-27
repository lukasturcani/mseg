import ruptures as rpt
import streamlit as st

from mseg._internal.utils import parse_data_file


def main() -> None:
    st.title("Ruptures Breakpoint Detection")
    data_file = st.file_uploader("Choose a file")
    if data_file is None:
        return
    data_df = parse_data_file(data_file.getvalue().decode()).sort("time")
    with st.sidebar:
        st.header("Parameters")
        breakpoints = rpt.Pelt(
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
                value=2000,
                help="penalty value",
            ),
        )


if __name__ == "__main__":
    main()
