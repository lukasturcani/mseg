import streamlit as st

from mseg._internal.scripts import (
    permutation_anova_app,
    rbeast_app,
    ruptures_app,
)


def main() -> None:
    pg = st.navigation(
        [
            st.Page(rbeast_app.__file__),
            st.Page(ruptures_app.__file__),
            st.Page(permutation_anova_app.__file__),
        ]
    )
    pg.run()


if __name__ == "__main__":
    main()
