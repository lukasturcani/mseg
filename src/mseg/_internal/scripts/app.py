import streamlit as st

from mseg._internal.utils import parse_data_file


def main() -> None:
    st.title("Segmentation Analysis")
    data_file = st.file_uploader("Choose a file")
    if data_file is None:
        return
    data_df = parse_data_file(data_file.getvalue().decode())
    st.write(data_df)


if __name__ == "__main__":
    main()
