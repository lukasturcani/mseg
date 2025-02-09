import pandas as pd
import seaborn as sns

sns.set_theme()
sns.set_color_codes()

from clasp.annotation.clasp import (
    ClaSPSegmentation,
    find_dominant_window_sizes,
)
from clasp.annotation.plotting.utils import (
    plot_time_series_with_change_points,
    plot_time_series_with_profiles,
)
from sktime.datasets import load_electric_devices_segmentation

ts, period_size, true_cps = load_electric_devices_segmentation()
_ = plot_time_series_with_change_points("Electric Devices", ts, true_cps)

clasp = ClaSPSegmentation(period_length=period_size, n_cps=5, fmt="sparse")
found_cps = clasp.fit_predict(ts)
profiles = clasp.profiles
scores = clasp.scores
print("The found change points are", found_cps.to_numpy())

_ = plot_time_series_with_profiles(
    "Electric Devices",
    ts,
    profiles,
    true_cps,
    found_cps,
)

clasp = ClaSPSegmentation(period_length=period_size, n_cps=5, fmt="dense")
found_segmentation = clasp.fit_predict(ts)
print(found_segmentation)

dominant_period_size = find_dominant_window_sizes(ts)
print("Dominant Period", dominant_period_size)

clasp = ClaSPSegmentation(period_length=dominant_period_size, n_cps=5)
found_cps = clasp.fit_predict(ts)
profiles = clasp.profiles
scores = clasp.scores

_ = plot_time_series_with_profiles(
    "ElectricDevices",
    ts,
    profiles,
    true_cps,
    found_cps,
)
