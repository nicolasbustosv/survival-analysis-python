"""Round-trip tests for figure and Excel export."""
import pytest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from survival_analysis.io import save_figure, write_xlsx


@pytest.fixture
def simple_fig():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 2])
    yield fig
    plt.close(fig)


def test_save_figure_png(simple_fig, tmp_path):
    basename = tmp_path / "test_fig"
    save_figure(simple_fig, basename, formats=("png",))
    assert (tmp_path / "test_fig.png").exists()


def test_save_figure_svg(simple_fig, tmp_path):
    basename = tmp_path / "test_fig"
    save_figure(simple_fig, basename, formats=("svg",))
    assert (tmp_path / "test_fig.svg").exists()


def test_write_xlsx_creates_file(tmp_path):
    tables = {
        "sheet1": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        "sheet2": pd.DataFrame({"x": [5, 6]}),
    }
    out = tmp_path / "test_out.xlsx"
    write_xlsx(tables, out)
    assert out.exists()


def test_write_xlsx_round_trip(tmp_path):
    df = pd.DataFrame({"hr": [1.2, 0.8], "p": [0.01, 0.5]})
    out = tmp_path / "round_trip.xlsx"
    write_xlsx({"results": df}, out)
    read_back = pd.read_excel(out, sheet_name="results")
    assert list(read_back.columns) == ["hr", "p"]
    assert len(read_back) == 2
