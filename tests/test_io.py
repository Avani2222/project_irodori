## test_io.py
import pandas as pd
import numpy as np
import pytest

from irodori.core import HyperTable
from irodori.io import load_csv, save_csv


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def sample_hyper_table():
    wavelengths = np.array([500, 600, 700])
    data = pd.DataFrame({
        500: [0.1, 0.2, 0.3],
        600: [0.4, 0.5, 0.6],
        700: [0.7, 0.8, 0.9],
    })

    # Add labels as a column
    data.insert(0, "label", ["A", "B", "C"])
    df["Label"] = labels
    ht = HyperTable(data, wavelengths=wavelengths, metadata={"source": "test"})
    return ht


@pytest.fixture
def sample_csv_file(tmp_path):
    """Creates a temporary CSV file with labels + 3 bands."""
    filepath = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        "Label": ["A", "B"],
        500: [0.1, 0.2],
        600: [0.3, 0.4],
        700: [0.5, 0.6],
    })
    df.to_csv(filepath, index=False)
    return filepath


# -----------------------------
# Tests for load_csv
# -----------------------------
def test_load_csv_with_header(sample_csv_file):
    ht = load_csv(sample_csv_file, wavelengths=[500, 600, 700], metadata={"sensor": "mock"})
    assert isinstance(ht, HyperTable)
    assert ht.samples == 2
    assert ht.bands == 3
    assert list(ht.labels) == ["A", "B"]
    assert np.allclose(ht.data.iloc[0].values, [0.1, 0.3, 0.5])
    assert np.all(ht.wavelengths == [500, 600, 700])
    assert ht.metadata["sensor"] == "mock"


def test_load_csv_without_header(tmp_path):
    filepath = tmp_path / "test_no_header.csv"
    df = pd.DataFrame([
        ["A", 0.1, 0.2, 0.3],
        ["B", 0.4, 0.5, 0.6],
    ])
    df.to_csv(filepath, index=False, header=False)

    ht = load_csv(filepath, wavelengths=[500, 600, 700], header=False)
    assert ht.samples == 2
    assert ht.bands == 3
    assert ht.labels[0] == "A"


# -----------------------------
# Tests for save_csv
# -----------------------------
def test_save_csv_roundtrip(sample_hyper_table, tmp_path):
    filepath = tmp_path / "roundtrip.csv"
    save_csv(sample_hyper_table, filepath)

    reloaded = pd.read_csv(filepath)
    assert "Label" in reloaded.columns
    assert reloaded.shape == (3, 4)  # 3 samples × (label + 3 bands)
    assert reloaded["Label"].tolist() == ["A", "B", "C"]


def test_save_csv_without_header(sample_hyper_table, tmp_path):
    filepath = tmp_path / "no_header.csv"
    save_csv(sample_hyper_table, filepath, include_header=False)

    reloaded = pd.read_csv(filepath, header=None)
    assert reloaded.shape == (3, 4)  # label + 3 bands
    assert reloaded.iloc[0, 0] == "A"


def test_save_csv_empty_data(tmp_path):
    empty_ht = HyperTable(pd.DataFrame(), wavelengths=None, labels=[])
    with pytest.raises(ValueError):
        save_csv(empty_ht, tmp_path / "bad.csv")
