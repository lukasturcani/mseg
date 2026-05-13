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


def parse_tabular_data_file(content: str) -> pl.DataFrame:
    """Read a wide-format file with subjects in column 1 and conditions after.

    The first row is a header. Column 1 is the subject identifier
    (e.g. ``rat``) and the remaining columns are condition names. Each
    subsequent row contains the subject ID followed by one numeric value per
    condition. Blank lines and rows with missing values are skipped.
    """
    rows = [line.split() for line in content.splitlines()]
    rows = [row for row in rows if row]
    if not rows:
        return pl.DataFrame()

    header = rows[0]
    subject_col = header[0]
    condition_cols = header[1:]
    n_cols = len(header)

    subjects: list[str] = []
    data: dict[str, list[float]] = {col: [] for col in condition_cols}
    for row in rows[1:]:
        if len(row) != n_cols:
            continue
        subjects.append(row[0])
        for col, value in zip(condition_cols, row[1:], strict=True):
            data[col].append(float(value))
    return pl.DataFrame({subject_col: subjects, **data})


def scatter_plot(data: pl.DataFrame) -> None:
    """Plot data."""
    fig = px.scatter(data, x="time", y="power")
    fig.show()


def line_plot(data: pl.DataFrame) -> None:
    """Plot data."""
    fig = px.line(data, x="time", y="power")
    fig.show()
