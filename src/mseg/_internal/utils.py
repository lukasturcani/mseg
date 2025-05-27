import plotly.express as px
import polars as pl


def parse_data_file(content: str) -> pl.DataFrame:
    """Read data from a path."""
    lines = content.splitlines()
    xs = []
    ys = []
    for line in lines:
        x, y = line.split()
        xs.append(float(x))
        ys.append(float(y))
    return pl.DataFrame({"time": xs, "power": ys})


def scatter_plot(data: pl.DataFrame) -> None:
    """Plot data."""
    fig = px.scatter(data, x="time", y="power")
    fig.show()


def line_plot(data: pl.DataFrame) -> None:
    """Plot data."""
    fig = px.line(data, x="time", y="power")
    fig.show()
