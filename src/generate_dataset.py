from pathlib import Path
import time

import pandas as pd

from xfoil_runner import run_xfoil


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)


# ---------------------------------------------------------
# NACA geometry
# ---------------------------------------------------------

def get_naca_features(naca):
    """
    Convert a NACA 4-digit code into geometry parameters.

    Example
    -------
    NACA 2412:

    maximum camber = 0.02
    camber position = 0.40
    thickness = 0.12

    Parameters
    ----------
    naca : str
        Four-digit NACA airfoil code.

    Returns
    -------
    tuple
        Maximum camber, camber position and thickness.
    """

    naca = str(naca)

    if len(naca) != 4 or not naca.isdigit():
        raise ValueError(
            f"Invalid NACA 4-digit code: {naca}"
        )

    max_camber = int(naca[0]) / 100
    camber_position = int(naca[1]) / 10
    thickness = int(naca[2:4]) / 100

    # Symmetric airfoils have no meaningful
    # camber position.
    if max_camber == 0:
        camber_position = 0.0

    return (
        max_camber,
        camber_position,
        thickness,
    )


# ---------------------------------------------------------
# XFOIL polar reader
# ---------------------------------------------------------

def load_xfoil_polar(file_path):
    """
    Read an XFOIL polar text file into a DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to an XFOIL polar output file.

    Returns
    -------
    pandas.DataFrame
        Clean aerodynamic operating-point data.
    """

    with open(file_path, "r") as file:
        lines = file.readlines()

    header_index = None

    for i, line in enumerate(lines):
        if (
            "alpha" in line
            and "CL" in line
            and "CD" in line
        ):
            header_index = i
            break

    if header_index is None:
        raise ValueError(
            f"Could not locate XFOIL header "
            f"in {file_path}"
        )

    data = pd.read_csv(
        file_path,
        sep=r"\s+",
        skiprows=header_index + 2,
        names=[
            "alpha",
            "CL",
            "CD",
            "CDp",
            "CM",
            "Top_Xtr",
            "Bot_Xtr",
        ],
    )

    # Convert values to numeric.
    # Invalid rows become NaN.
    data = data.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # CL, CD and alpha are essential.
    data = data.dropna(
        subset=[
            "alpha",
            "CL",
            "CD",
        ]
    )

    return data


# ---------------------------------------------------------
# Run XFOIL with automatic retry
# ---------------------------------------------------------

def run_with_retry(
    naca,
    reynolds,
    alpha_start,
    alpha_end,
    alpha_step,
    max_attempts=2,
):
    """
    Run XFOIL and retry if the entire process fails.

    Individual non-converged angles are retained as
    missing operating points. A retry is only performed
    when the complete XFOIL process fails.

    Parameters
    ----------
    naca : str
        NACA 4-digit airfoil code.
    reynolds : int
        Reynolds number.
    alpha_start : float
        Minimum angle of attack.
    alpha_end : float
        Maximum angle of attack.
    alpha_step : float
        Angle increment.
    max_attempts : int
        Maximum number of complete XFOIL attempts.

    Returns
    -------
    tuple
        Polar file path and number of attempts used.
    """

    last_error = None

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        try:

            print(
                f"Attempt {attempt}/{max_attempts}"
            )

            polar_file = run_xfoil(
                naca=naca,
                reynolds=reynolds,
                alpha_start=alpha_start,
                alpha_end=alpha_end,
                alpha_step=alpha_step,
            )

            return polar_file, attempt

        except Exception as error:

            last_error = error

            print()
            print(
                f"Attempt {attempt} failed:"
            )
            print(error)

            if attempt < max_attempts:

                print(
                    "Retrying XFOIL run..."
                )

                # Brief pause to ensure the previous
                # XFOIL process has fully terminated.
                time.sleep(2)

    raise RuntimeError(
        f"XFOIL failed after {max_attempts} "
        f"attempts for NACA {naca}, "
        f"Re = {reynolds}. "
        f"Final error: {last_error}"
    )


# ---------------------------------------------------------
# Final dataset generation
# ---------------------------------------------------------

def generate_final_dataset():
    """
    Generate the final aerodynamic dataset for the
    MMA3001 airfoil surrogate-model project.
    """

    # -----------------------------------------------------
    # Final NACA geometry grid
    # -----------------------------------------------------

    airfoils = [

        # Symmetric airfoils
        "0009",
        "0012",
        "0015",

        # 2% maximum camber, p = 0.2
        "2209",
        "2212",
        "2215",

        # 2% maximum camber, p = 0.4
        "2409",
        "2412",
        "2415",

        # 2% maximum camber, p = 0.6
        "2609",
        "2612",
        "2615",

        # 4% maximum camber, p = 0.2
        "4209",
        "4212",
        "4215",

        # 4% maximum camber, p = 0.4
        "4409",
        "4412",
        "4415",

        # 4% maximum camber, p = 0.6
        "4609",
        "4612",
        "4615",
    ]

    reynolds_numbers = [
        100000,
        200000,
        300000,
        400000,
        500000,
    ]

    alpha_start = -6
    alpha_end = 12
    alpha_step = 1

    expected_points_per_run = 19

    all_data = []
    summary_rows = []

    total_runs = (
        len(airfoils)
        * len(reynolds_numbers)
    )

    total_expected_points = (
        total_runs
        * expected_points_per_run
    )

    run_number = 0

    print()
    print(
        "=========================================="
    )
    print(
        "MMA3001 FINAL XFOIL DATASET GENERATOR"
    )
    print(
        "=========================================="
    )

    print()
    print(
        f"Airfoils: {len(airfoils)}"
    )

    print(
        f"Reynolds numbers: "
        f"{len(reynolds_numbers)}"
    )

    print(
        f"Total XFOIL runs: "
        f"{total_runs}"
    )

    print(
        f"Maximum aerodynamic points: "
        f"{total_expected_points}"
    )

    # -----------------------------------------------------
    # Run every airfoil / Reynolds combination
    # -----------------------------------------------------

    for naca in airfoils:

        (
            max_camber,
            camber_position,
            thickness,
        ) = get_naca_features(naca)

        for reynolds in reynolds_numbers:

            run_number += 1

            print()
            print(
                "=========================================="
            )

            print(
                f"RUN {run_number}/{total_runs}"
            )

            print(
                f"NACA {naca} | "
                f"Re = {reynolds}"
            )

            print(
                "=========================================="
            )

            try:

                (
                    polar_file,
                    attempts_used,
                ) = run_with_retry(
                    naca=naca,
                    reynolds=reynolds,
                    alpha_start=alpha_start,
                    alpha_end=alpha_end,
                    alpha_step=alpha_step,
                    max_attempts=2,
                )

                polar = load_xfoil_polar(
                    polar_file
                )

                converged_points = len(
                    polar
                )

                coverage = (
                    converged_points
                    / expected_points_per_run
                    * 100
                )

                # -----------------------------------------
                # Add engineering input variables
                # -----------------------------------------

                polar["airfoil"] = (
                    f"NACA {naca}"
                )

                polar["max_camber"] = (
                    max_camber
                )

                polar["camber_position"] = (
                    camber_position
                )

                polar["thickness"] = (
                    thickness
                )

                polar["reynolds"] = (
                    reynolds
                )

                all_data.append(
                    polar
                )

                summary_rows.append(
                    {
                        "airfoil":
                            f"NACA {naca}",

                        "reynolds":
                            reynolds,

                        "expected_points":
                            expected_points_per_run,

                        "converged_points":
                            converged_points,

                        "coverage_percent":
                            coverage,

                        "attempts_used":
                            attempts_used,

                        "status":
                            "success",
                    }
                )

                print()
                print(
                    f"Converged points: "
                    f"{converged_points}/"
                    f"{expected_points_per_run}"
                )

                print(
                    f"Coverage: "
                    f"{coverage:.1f}%"
                )

            except Exception as error:

                print()
                print(
                    "RUN FAILED AFTER RETRY"
                )

                print(error)

                summary_rows.append(
                    {
                        "airfoil":
                            f"NACA {naca}",

                        "reynolds":
                            reynolds,

                        "expected_points":
                            expected_points_per_run,

                        "converged_points":
                            0,

                        "coverage_percent":
                            0.0,

                        "attempts_used":
                            2,

                        "status":
                            f"failed: {error}",
                    }
                )

    # -----------------------------------------------------
    # Combine successful aerodynamic data
    # -----------------------------------------------------

    if not all_data:

        raise RuntimeError(
            "No XFOIL simulations produced "
            "usable aerodynamic data."
        )

    dataset = pd.concat(
        all_data,
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Arrange columns
    # -----------------------------------------------------

    column_order = [
        "airfoil",
        "max_camber",
        "camber_position",
        "thickness",
        "reynolds",
        "alpha",
        "CL",
        "CD",
        "CDp",
        "CM",
        "Top_Xtr",
        "Bot_Xtr",
    ]

    dataset = dataset[
        column_order
    ]

    # Sort into a consistent order.
    dataset = dataset.sort_values(
        by=[
            "max_camber",
            "camber_position",
            "thickness",
            "reynolds",
            "alpha",
        ]
    ).reset_index(
        drop=True
    )

    summary = pd.DataFrame(
        summary_rows
    )

    # -----------------------------------------------------
    # Save final outputs
    # -----------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = (
        PROCESSED_DATA_DIR
        / "airfoil_dataset_final.csv"
    )

    summary_path = (
        PROCESSED_DATA_DIR
        / "generation_summary_final.csv"
    )

    dataset.to_csv(
        dataset_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # -----------------------------------------------------
    # Final generation statistics
    # -----------------------------------------------------

    total_obtained = len(
        dataset
    )

    overall_coverage = (
        total_obtained
        / total_expected_points
        * 100
    )

    failed_runs = (
        summary["status"]
        .str.startswith("failed")
        .sum()
    )

    partial_runs = (
        (
            summary["converged_points"]
            > 0
        )
        &
        (
            summary["converged_points"]
            < expected_points_per_run
        )
    ).sum()

    full_runs = (
        summary["converged_points"]
        == expected_points_per_run
    ).sum()

    print()
    print(
        "=========================================="
    )

    print(
        "FINAL DATASET COMPLETE"
    )

    print(
        "=========================================="
    )

    print()

    print(
        f"Expected aerodynamic points: "
        f"{total_expected_points}"
    )

    print(
        f"Obtained aerodynamic points: "
        f"{total_obtained}"
    )

    print(
        f"Overall coverage: "
        f"{overall_coverage:.2f}%"
    )

    print()

    print(
        f"Complete XFOIL runs: "
        f"{full_runs}"
    )

    print(
        f"Partial XFOIL runs: "
        f"{partial_runs}"
    )

    print(
        f"Failed XFOIL runs: "
        f"{failed_runs}"
    )

    print()

    print(
        "Dataset saved to:"
    )

    print(
        dataset_path
    )

    print()

    print(
        "Generation summary saved to:"
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    generate_final_dataset()