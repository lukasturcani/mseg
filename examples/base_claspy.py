import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import ruptures as rpt
from claspy.segmentation import BinaryClaSPSegmentation

# Generate synthetic data using Polars
np.random.seed(42)
n = 300
df = pl.DataFrame(
    {
        "x": np.arange(200),
        "y": np.concatenate(
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
        ),
    }
)

# Convert 'y' column to NumPy array for ruptures
signal = df["y"].to_numpy()

clasp = BinaryClaSPSegmentation()
indices = clasp.fit_predict(signal)
change_points = df["x"][indices]
print(indices, change_points)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df["x"], df["y"], label="Time Series", color="blue")
for cp in change_points:
    plt.axvline(
        x=cp, color="red", linestyle="--", label="Detected Change Point"
    )
plt.xlabel("X (Time or Index)")
plt.ylabel("Y (Time Series Value)")
plt.legend()
plt.title("Automatic Change Point Detection using Claspy")
plt.show()
