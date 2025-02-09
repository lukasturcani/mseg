import numpy as np
import polars as pl
from pyearth import Earth
from matplotlib import pyplot
from itertools import islice

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

# Prepare data for Earth model
X = (
    df["time"].to_numpy().reshape(-1, 1)
)  # reshape to 2D array as required by Earth
y = df["value"].to_numpy()

# Fit an Earth model
model = Earth()
model.fit(X, y)

# Print the model
print(model.trace())
print(model.summary())

knots = [
    bf.get_knot()
    for bf in model.basis_
    if hasattr(bf, "get_knot") and not bf.is_pruned()
]
print(f"knots: {knots}")

# Plot the model
y_hat = model.predict(X)
pyplot.figure()
pyplot.plot(X, y, label="Actual")
pyplot.plot(X, y_hat, "b.", label="Predicted")
pyplot.xlabel("Time")
pyplot.ylabel("Value")
pyplot.title("Earth Model Time Series Fit")
pyplot.legend()
pyplot.show()

basis_functions = model.basis_
coefficients = model.coef_
