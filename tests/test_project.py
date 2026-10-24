from pathlib import Path
import sys

import pandas as pd
import pytest


# ---------------------------------------------------------
# Allow tests to import source code
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR),
)


from generate_dataset import (
    get_naca_features,
    load_xfoil_polar,
)


# ---------------------------------------------------------
# NACA geometry tests
# ---------------------------------------------------------

def test_naca_2412_features():
    """
    Check that NACA 2412 is converted into the
    correct geometric input variables.
    """

    max_camber, position, thickness = (
        get_naca_features("2412")
    )

    assert max_camber == pytest.approx(0.02)
    assert position == pytest.approx(0.4)
    assert thickness == pytest.approx(0.12)


def test_symmetric_airfoil_features():
    """
    Symmetric NACA airfoils should have zero
    camber and zero camber position.
    """

    max_camber, position, thickness = (
        get_naca_features("0012")
    )

    assert max_camber == pytest.approx(0.0)
    assert position == pytest.approx(0.0)
    assert thickness == pytest.approx(0.12)


def test_invalid_naca_code():
    """
    Invalid NACA codes should raise ValueError.
    """

    with pytest.raises(ValueError):

        get_naca_features("ABC")


# ---------------------------------------------------------
# Final dataset tests
# ---------------------------------------------------------

def test_final_dataset_exists():
    """
    Check that the final aerodynamic dataset exists.
    """

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "airfoil_dataset_final.csv"
    )

    assert dataset_path.exists()


def test_final_dataset_columns():
    """
    Check required ML input/output columns.
    """

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "airfoil_dataset_final.csv"
    )

    data = pd.read_csv(
        dataset_path
    )

    required_columns = {
        "airfoil",
        "max_camber",
        "camber_position",
        "thickness",
        "reynolds",
        "alpha",
        "CL",
        "CD",
    }

    assert required_columns.issubset(
        data.columns
    )


def test_no_missing_model_values():
    """
    Model inputs and outputs must not contain
    missing values.
    """

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "airfoil_dataset_final.csv"
    )

    data = pd.read_csv(
        dataset_path
    )

    model_columns = [
        "max_camber",
        "camber_position",
        "thickness",
        "reynolds",
        "alpha",
        "CL",
        "CD",
    ]

    assert (
        data[model_columns]
        .isna()
        .sum()
        .sum()
        == 0
    )


def test_no_duplicate_rows():
    """
    Final aerodynamic dataset should not contain
    duplicate operating points.
    """

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "airfoil_dataset_final.csv"
    )

    data = pd.read_csv(
        dataset_path
    )

    assert data.duplicated().sum() == 0


def test_input_domain():
    """
    Check that aerodynamic operating points remain
    inside the intended project domain.
    """

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "airfoil_dataset_final.csv"
    )

    data = pd.read_csv(
        dataset_path
    )

    assert data["alpha"].between(
        -6,
        12,
    ).all()

    assert data["reynolds"].between(
        100000,
        500000,
    ).all()

    assert data["max_camber"].between(
        0.0,
        0.04,
    ).all()

    assert data["thickness"].between(
        0.09,
        0.15,
    ).all()