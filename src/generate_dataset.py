from pathlib import Path

import pandas as pd

from xfoil_runner import run_xfoil


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


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
    """

    naca = str(naca)

    if len(naca) != 4 or not naca.isdigit():
        raise ValueError(
            f"Invalid NACA 4-digit code: {naca}"
        )

    max_camber = int(naca[0]) / 100
    camber_position = int(naca[1]) / 10
    thickness = int(naca[2:4]) / 100

    return max_camber, camber_position, thickness


# ---------------------------------------------------------
# XFOIL polar reader
# ---------------------------------------------------------

def load_xfoil_polar(file_path):
    """
    Read an XFOIL polar text file into a DataFrame.
    """

    with open(file_path, "r") as f:
        lines = f.readlines()

    header_index = None

    for i, line in enumerate(lines):
        if "alpha" in line and "CL" in line and "CD" in line:
            header_index = i
            break

    if header_index is None:
        raise ValueError(
            f"Could not locate XFOIL data header in {file_path}"
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

    # Convert everything to numbers.
    # Invalid rows become NaN.
    data = data.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # CL and CD are essential for this project.
    data = data.dropna(
        subset=["alpha", "CL", "CD"]
    )

    return data


# ---------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------

def generate_pilot_dataset():
    """
    Generate a small pilot aerodynamic dataset.

    This is used to verify the full data-generation
    workflow before generating the final large dataset.
    """

    airfoils = [
        "0009",
        "0015",
        "2212",
        "2412",
        "2612",
        "4415",
        "6412",
    ]

    reynolds_numbers = [
        100000,
        200000,
        500000,
    ]

    alpha_start = -6
    alpha_end = 12
    alpha_step = 1

    expected_points = 19

    all_data = []
    summary_rows = []

    total_runs = (
        len(airfoils)
        * len(reynolds_numbers)
    )

    run_number = 0

    print()
    print("======================================")
    print("MMA3001 XFOIL PILOT DATASET GENERATOR")
    print("======================================")
    print()
    print(f"Airfoils: {len(airfoils)}")
    print(
        f"Reynolds numbers: "
        f"{len(reynolds_numbers)}"
    )
    print(f"Total XFOIL runs: {total_runs}")
    print()

    for naca in airfoils:

        max_camber, camber_position, thickness = (
            get_naca_features(naca)
        )

        for reynolds in reynolds_numbers:

            run_number += 1

            print()
            print(
                f"RUN {run_number}/{total_runs}"
            )
            print(
                f"NACA {naca} | Re = {reynolds}"
            )

            try:
                polar_file = run_xfoil(
                    naca=naca,
                    reynolds=reynolds,
                    alpha_start=alpha_start,
                    alpha_end=alpha_end,
                    alpha_step=alpha_step,
                )

                polar = load_xfoil_polar(
                    polar_file
                )

                converged_points = len(polar)

                coverage = (
                    converged_points
                    / expected_points
                    * 100
                )

                # Add engineering input variables
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

                all_data.append(polar)

                summary_rows.append(
                    {
                        "airfoil": f"NACA {naca}",
                        "reynolds": reynolds,
                        "expected_points":
                            expected_points,
                        "converged_points":
                            converged_points,
                        "coverage_percent":
                            coverage,
                        "status": "success",
                    }
                )

                print(
                    f"Converged points: "
                    f"{converged_points}/"
                    f"{expected_points}"
                )

                print(
                    f"Coverage: "
                    f"{coverage:.1f}%"
                )

            except Exception as error:

                print(
                    f"RUN FAILED: {error}"
                )

                summary_rows.append(
                    {
                        "airfoil":
                            f"NACA {naca}",
                        "reynolds":
                            reynolds,
                        "expected_points":
                            expected_points,
                        "converged_points":
                            0,
                        "coverage_percent":
                            0,
                        "status":
                            f"failed: {error}",
                    }
                )

    # -----------------------------------------------------
    # Combine all successful aerodynamic data
    # -----------------------------------------------------

    if not all_data:
        raise RuntimeError(
            "No XFOIL runs produced usable data."
        )

    dataset = pd.concat(
        all_data,
        ignore_index=True,
    )

    # Put important ML inputs/outputs first
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

    dataset = dataset[column_order]

    summary = pd.DataFrame(
        summary_rows
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = (
        PROCESSED_DATA_DIR
        / "airfoil_dataset_pilot.csv"
    )

    summary_path = (
        PROCESSED_DATA_DIR
        / "generation_summary_pilot.csv"
    )

    dataset.to_csv(
        dataset_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("======================================")
    print("PILOT DATASET COMPLETE")
    print("======================================")
    print()

    print(
        f"Total aerodynamic rows: "
        f"{len(dataset)}"
    )

    print(
        f"Dataset saved to:\n"
        f"{dataset_path}"
    )

    print()

    print(
        f"Generation summary saved to:\n"
        f"{summary_path}"
    )


if __name__ == "__main__":
    generate_pilot_dataset()