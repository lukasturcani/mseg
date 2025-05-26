from dataclasses import dataclass
from typing import Protocol

import altair as alt
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

    chart = alt.Chart(data_df).mark_line().encode(x="time", y="power")
    st.altair_chart(chart)

    with st.sidebar:
        st.header("Parameters")
        season = st.selectbox(
            "season",
            [
                "none",
                "harmonic",
                "dummy",
                "svd",
            ],
            index=0,
            help=(
                """
                * none - trend-only data with no seasonality
                * harmonic - the seasonal/peridoic component modelled via harmonic curves
                * dummy - the seasonal component  modelled via a dummy basis (i.e., pulse-like bases)
                * svd - svd-derived bases (experimental feature)
                """  # noqa: E501
            ),
        )
        has_outlier = st.checkbox(
            "hasOutlier",
            help=(
                """
                    if true, the model with an outlier component will be fitted (if season is 'none' then
                    Y=trend+outlier+error, or if season is not 'none' then Y=trend+season+outlier+error).
                    """  # noqa: E501
            ),
        )
        raw_output = rb.beast(
            Y=data_df["power"].to_numpy(),
            start=st.number_input(
                "start",
                value=data_df.select(pl.col("time").min()).item(),
                help="The start time of the regular time series.",
            ),
            deltat=st.number_input(
                "deltat",
                value=delta_t,
                help="The time interval between consecutive datapoints.",
            ),
            season=season,
            period=st.number_input(
                "period",
                value=float("nan"),
                help=(
                    """
                    a number to specify the period if periodic/seasonal variations
                    are present in the data. If period is given a zero, negative value or 'none'
                    it suggests no seasonal/periodic component in the signal. (season='none'
                    also suggests no periodic component).
                    The unit of 'period', if any, should be consistent with the unit of 'deltat'.
                    """  # noqa: E501
                ),
            ),
            scp_minmax=st.slider(
                "scp_minmax",
                min_value=0,
                max_value=50,
                step=1,
                value=(0, 10),
                help=(
                    "the min and max number of seasonal changepoints allowed"
                ),
            ),
            sorder_minmax=st.slider(
                "sorder_minmax",
                min_value=0,
                max_value=10,
                step=1,
                value=(0, 5),
                help=(
                    "the min and max harmonic orders of seasonal "
                    "changepoints (scp) allowed"
                ),
            ),
            sseg_minlength=st.number_input(
                "sseg_minlength",
                value=None,
                help=(
                    "the min length of the segment for the seasonal component "
                    "i.e., the min distance between neighorbing changepoints)"
                ),
            ),
            sseg_leftmargin=st.number_input(
                "sseg_leftmargin",
                value=None,
                help=(
                    """
                    the number of leftmost data points excluded for seasonal changepoint detection.
                    That is,  no changepoints are allowed in the starting window/segment of length sseg_leftmargin.
                    sseg_leftmargin must be an unitless integer - the number of time intervals/data points so that the
                    time window in the original unit is sseg_leftmargin*deltat. If missing, sseg_leftmargin defaults
                    to the minimum segment length 'sseg_min'
                    """  # noqa: E501
                ),
            ),
            sseg_rightmargin=st.number_input(
                "sseg_rightmargin",
                value=None,
                help=(
                    """
                    the number of rightmost data points excluded for seasonal changepoint detection.
                    That is,  no changepoints are allowed in the ending window/segment of length sseg_rightmargin.
                    sseg_rightmargin must be an unitless integer - the number of time intervals/data points so that the
                    time window in the original unit is sseg_rightmargin*deltat. If missing, sseg_rightmargin defaults
                    to the minimum segment length 'sseg_min'
                    """  # noqa: E501
                ),
            ),
            tcp_minmax=st.slider(
                "tcp_minmax",
                min_value=0,
                max_value=50,
                step=1,
                value=(0, 10),
                help=(
                    "the min and max numbers of trend changepoints (tcp) "
                    "allowed"
                ),
            ),
            torder_minmax=st.slider(
                "torder_minmax",
                min_value=0,
                max_value=10,
                step=1,
                value=(0, 1),
                help=(
                    "the min and max orders of polynomials used "
                    "to model the trend"
                ),
            ),
            tseg_minlength=st.number_input(
                "tseg_minlength",
                value=None,
                help=(
                    """
                    the min length of the segment for the trend component (i.e.,
                    the min distance between neighorbing changepoints)
                    """  # noqa: E501
                ),
            ),
            tseg_leftmargin=st.number_input(
                "tseg_leftmargin",
                value=None,
                help=(
                    """
                    the number of leftmost data points excluded for trend changepoint detection.
                    That is,  no trend changepoints are allowed in the starting window/segment of length tseg_leftmargin.
                    tseg_leftmargin must be an unitless integer - the number of time intervals/data points so that the
                    time window in the original unit is tseg_leftmargin*deltat.
                    """  # noqa: E501
                ),
            ),
            tseg_rightmargin=st.number_input(
                "tseg_rightmargin",
                value=None,
                help=(
                    """
                    the number of rightmost data points excluded for trend changepoint detection.
                    That is,  no trend changepoints are allowed in the ending window/segment of length tseg_rightmargin.
                    tseg_rightmargin must be an unitless integer-the number of time intervals/data points so that the
                    time window in the original unit is tseg_rightmargin*deltat.
                    """  # noqa: E501
                ),
            ),
            method=st.selectbox(
                "method",
                ["bayes", "bic", "aic", "aicc", "hic", "bic0.25", "bic0.5"],
                index=0,
                help=(
                    """
                    method to formulat model posterior probability. Possible values are:
                    * bayes - the full Bayesian formulation (this is the default)
                    * bic -  approximation of posterior probability using the Bayesian information criterion (bic)
                    * aic -  approximation of posterior probability using the Akaike information criterion (aic)
                    * aicc - approximation of posterior probability using the corrected Akaike information criterion (aicc)
                    * hic - approximation of  posterior probability using the Hannan-Quinn information criterion  (hic)
                    * bic0.25 - approximation using the Bayesian information criterion adopted from Kim et al. (2016) <doi: 10.1016/j.jspi.2015.09.008>; bic0.25=n*ln(SSE)+0.25k*ln(n) with less complexity penelaty than the standard BIC.
                    * bic0.50 - the same as above except that the penalty factor is 0.50.
                    * bic1.5 - the same as above except that the penalty factor is 1.5.
                    * bic2 - the same as above except that the penalty factor is 2.0.
                    """  # noqa: E501
                ),
            ),
            detrend=st.checkbox(
                "detrend",
                help=(
                    """
                    if true, the input time series will be first de-trend before applying
                    beast by removing a global trend
                    """  # noqa: E501
                ),
            ),
            deseasonalize=st.checkbox(
                "deseasonalize",
                help=(
                    """
                    if true, the input time series will be first de-seasonalized before applying
                    beast by removing a global seasonal component
                    """  # noqa: E501
                ),
            ),
            mcmc_seed=st.number_input(
                "mcmc_seed",
                value=1,
                help=(
                    """
                    a seed for the random number generator; set it to a non-zero integer to
                    reproduce the results among different runs
                    """  # noqa: E501
                ),
            ),
            mcmc_burbin=st.number_input(
                "mcmc_burnin",
                value=200,
                help=(
                    """
                    the number of initial samples of each chain to be discarded
                    """
                ),
            ),
            mcmc_chains=st.number_input(
                "mcmc_chains",
                value=3,
                help=(
                    """
                    the number of MCMC chains; the larger, the better but with more computation.
                    """  # noqa: E501
                ),
            ),
            mcmc_thin=st.number_input(
                "mcmc_thin",
                value=5,
                help=(
                    """
                    a thinning factor for MCMC chains: take every 'mcmc.thin'-th sample
                    """  # noqa: E501
                ),
            ),
            mcmc_samples=st.number_input(
                "mcmc_samples",
                value=8000,
                help=(
                    """
                    number of MCMC samples collected; the larger, the better
                    """
                ),
            ),
            precValue=st.number_input(
                "precValue",
                value=1.5,
                min_value=0.0,
                help=(
                    """
                    numeric (>0); the hyperparameter of the precision prior; precValue
                    is useful only when precPriorType='constant', as further explained below
                    """  # noqa: E501
                ),
            ),
            precPriorType=st.selectbox(
                "precPriorType",
                ["componentwise", "uniform", "constant", "orderwise"],
                index=0,
                help=(
                    """
                    * constant - the precision parameter used to parameterize the model coefficients is fixed to
                    a const specified by precValue. In other words, precValue is a user-defined hyperparameter
                    and the fitting result may be sensitive to the chosen values of precValue.
                    * uniform - the precision parameter used to parameterize the model coefficients is a random variable;
                    its initial value is specified by precValue. In other words, precValue will be inferred by the MCMC,
                    so the fitting result will be insensitive to the chose inital value of precValue.
                    * componentwise - multiple precision parameters are used to parameterize the model coefficients for
                    individual components (e.g., one for season and another for trend); their initial values is specified
                    by precValue. In other words, precValue will be inferred by the MCMC, so the fitting result will be
                    insensitive to the choice in precValue.
                    * orderwise - multiple precision parameters are used to parameterize the model coefficients not just for
                    individual components but also for individual orders of each component; their initial values is specified
                    by precValue. In other words, precValue will be inferred by the MCMC, so the fitting result will be
                    insensitive to the choice in precValue.
                    """  # noqa: E501
                ),
            ),
            hasOutlier=has_outlier,
            ocp_minmax=st.slider(
                "ocp_minmax",
                min_value=0,
                max_value=50,
                step=1,
                value=(0, 10),
                help=(
                    """
                    the min and max numbers of outlier-type changepoints (ocp) allowed in the time series.
                    Ocp refers to spikes or dips at isolated times that can't be modeled as trends or seasonal terms.
                    """  # noqa: E501
                ),
            ),
            print_param=False,
            print_progress=False,
            print_warning=False,
            quiet=True,
            dump_ci=False,
        )
    output = RBeastOutput(
        trend=raw_output.trend,
    )
    if season != "none":
        output.season = raw_output.season
    if has_outlier:
        output.outlier = raw_output.outlier

    st.header("Trend Change Points")
    st.dataframe(_trend_change_points(output.trend))
    trend_cp_probability_chart = (
        alt.Chart(
            _trend_change_point_probability(raw_output.time, output.trend)
        )
        .mark_line(
            color="green",
        )
        .encode(
            x="time",
            y="probability",
        )
    )
    if st.checkbox("Show original data", value=True):
        trend_cp_probability_chart = alt.layer(
            chart, trend_cp_probability_chart
        ).resolve_scale(y="independent")

    st.altair_chart(trend_cp_probability_chart)
    if output.season is not None:
        st.header("Seasonal Change Points")
        st.dataframe(_seasonal_change_points(output.season))
    if output.outlier is not None:
        st.header("Outlier Change Points")
        st.dataframe(_outlier_change_points(output.outlier))


def _get_delta_t(data: pl.DataFrame, tolerance: float = 1e-3) -> float | None:
    diffs = data["time"].diff().drop_nulls()
    ref = diffs[0]
    all_close = np.all(np.isclose(diffs.to_numpy(), ref, atol=tolerance))
    if all_close:
        return ref
    return None


@dataclass(slots=True)
class ChangePoint:
    location: float
    probability: float


class TrendOutput(Protocol):
    cp: list[float]
    cpPr: list[float]  # noqa: N815
    cpOccPr: list[float]  # noqa: N815


class SeasonOputput(Protocol):
    cp: list[float]
    cpPr: list[float]  # noqa: N815
    cpOccPr: list[float]  # noqa: N815


class OutlierOutput(Protocol):
    cp: list[float]
    cpPr: list[float]  # noqa: N815
    cpOccPr: list[float]  # noqa: N815


@dataclass(slots=True)
class RBeastOutput:
    trend: TrendOutput
    season: SeasonOputput | None = None
    outlier: OutlierOutput | None = None


def _trend_change_points(
    trend: TrendOutput,
) -> pl.DataFrame:
    cp_df = pl.DataFrame(
        {
            "time": trend.cp,
            "probability": trend.cpPr,
        },
    ).sort("time")
    return cp_df.with_columns(
        pl.int_range(1, len(cp_df) + 1).alias("")
    ).select("", "time", "probability")


def _trend_change_point_probability(
    time: list[float],
    trend: TrendOutput,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": time,
            "probability": trend.cpOccPr,
        },
    ).sort("time")


def _seasonal_change_points(
    season: SeasonOputput,
) -> pl.DataFrame:
    cp_df = pl.DataFrame(
        {
            "time": season.cp,
            "probability": season.cpPr,
        },
    ).sort("time")
    return cp_df.with_columns(
        pl.int_range(1, len(cp_df) + 1).alias("")
    ).select("", "time", "probability")


def _outlier_change_points(
    outlier: OutlierOutput,
) -> pl.DataFrame:
    cp_df = pl.DataFrame(
        {
            "time": outlier.cp,
            "probability": outlier.cpPr,
        },
    ).sort("time")
    return cp_df.with_columns(
        pl.int_range(1, len(cp_df) + 1).alias("")
    ).select("", "time", "probability")


if __name__ == "__main__":
    main()
