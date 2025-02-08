import numpy
from pyearth import Earth
from matplotlib import pyplot
from itertools import islice

# Create some fake data
numpy.random.seed(0)
m = 1000
n = 10
X = 80 * numpy.random.uniform(size=(m, n)) - 40
y = numpy.abs(X[:, 6] - 4.0) + 1 * numpy.random.normal(size=m)

# Fit an Earth model
model = Earth()
model.fit(X, y)

# Print the model
print(model.trace())
print(model.summary())

knots = [bf.get_knot() for bf in list(model.basis_)[1:] if not bf.is_pruned()]
print(f"knots: {knots}")

# Plot the model
y_hat = model.predict(X)
pyplot.figure()
pyplot.plot(X[:, 6], y, "r.")
pyplot.plot(X[:, 6], y_hat, "b.")
pyplot.xlabel("x_6")
pyplot.ylabel("y")
pyplot.title("Simple Earth Example")
pyplot.show()

basis_functions = model.basis_
coefficients = model.coef_
