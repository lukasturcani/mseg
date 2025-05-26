import streamlit as st

from mseg._internal.utils import parse_data_file


def main() -> None:
    st.title("Segmentation Analysis")
    data_file = st.file_uploader("Choose a file")
    if data_file is None:
        return
    data_df = parse_data_file(data_file.getvalue().decode())
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

        st.header("Ruptures Parameters")


if __name__ == "__main__":
    main()
