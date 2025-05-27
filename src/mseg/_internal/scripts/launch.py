import subprocess
import sys
from pathlib import Path

from mseg._internal.scripts import app


def main() -> None:
    script_path = Path(app.__file__).resolve()
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "streamlit", "run", str(script_path)],
        check=True,
    )
