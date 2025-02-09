import polars as pl
import numpy as np
import stumpy  # Matrix Profile-based change point detection
import matplotlib.pyplot as plt

# Simulated time series with more structural breaks
times = np.arange(200)
data = np.concatenate(
    [
        np.random.normal(0, 1, 30),
        np.random.normal(5, 1, 30),
        np.random.normal(-3, 1, 30),
        np.random.normal(8, 1, 30),
        np.random.normal(2, 1, 40),
        np.random.normal(
            -5, 1, 40
        ),  # Change at t=30, t=60, t=90, t=120, t=160
    ]
)

df = pl.DataFrame({"time": times, "value": data})

# Apply moving average for smoothing
window_size = 5
df = df.with_columns(
    pl.col("value").rolling_mean(window_size).alias("smoothed_value")
)


# Function to detect breakpoints using STUMPY FLOSS with Corrected Arc Curve Peaks
def detect_breakpoints(
    df, column="smoothed_value", window_size=20, L_factor=5, n_breakpoints=5
):
    series = df[column].drop_nulls().to_numpy()
    matrix_profile = stumpy.stump(series, m=window_size)
    L = window_size * L_factor  # Increase L to improve segmentation
    floss_obj = stumpy.floss(matrix_profile, series, m=window_size, L=L)
    floss_scores = floss_obj.cac_1d_  # Get corrected arc curve scores

    # Identify peak locations in the CAC as breakpoints
    peak_indices = np.argsort(floss_scores)[-n_breakpoints:]
    return sorted(peak_indices)


# Detect breakpoints
breakpoints = detect_breakpoints(df)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(df["time"], df["smoothed_value"], label="Smoothed Time Series")
for bp in breakpoints:
    plt.axvline(x=bp, color="r", linestyle="--", label=f"Breakpoint at {bp}")
plt.legend()
plt.title("Anchored Time Series Chain with More Regime Changes")
plt.show()
