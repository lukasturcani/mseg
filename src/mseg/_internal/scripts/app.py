import streamlit as st

from mseg._internal.scripts import rbeast_app, ruptures_app


def main() -> None:
    pg = st.navigation(
        [st.Page(rbeast_app.__file__), st.Page(ruptures_app.__file__)]
    )
    pg.run()


if __name__ == "__main__":
    main()
