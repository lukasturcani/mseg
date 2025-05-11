import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import Rbeast as rb

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


output = rb.beast(signal, season="none")
rb.plot(output)
rb.print(output)
plt.show()
