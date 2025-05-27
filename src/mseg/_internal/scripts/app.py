import streamlit as st

from mseg._internal.scripts import rbeast, ruptures


def main() -> None:
    pg = st.navigation([st.Page(rbeast.__file__), st.Page(ruptures.__file__)])
    pg.run()


if __name__ == "__main__":
    main()
