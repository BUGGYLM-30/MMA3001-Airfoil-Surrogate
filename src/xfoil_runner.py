from pathlib import Path
import subprocess


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

XFOIL_EXE = Path(
    r"C:\Users\levi3\OneDrive\Desktop\xfoil.exe"
)


def run_xfoil(
    naca="2412",
    reynolds=200000,
    alpha_start=-6,
    alpha_end=12,
    alpha_step=1,
):
    """
    Run XFOIL for a NACA 4-digit airfoil.

    The angle-of-attack sweep begins at 0 degrees and
    moves toward positive angles first. XFOIL is then
    reinitialised before sweeping toward negative angles.
    This improves convergence compared with one continuous
    sweep from negative to positive angles.

    Parameters
    ----------
    naca : str
        Four-digit NACA airfoil code.
    reynolds : int
        Reynolds number.
    alpha_start : float
        Minimum angle of attack in degrees.
    alpha_end : float
        Maximum angle of attack in degrees.
    alpha_step : float
        Angle-of-attack increment in degrees.

    Returns
    -------
    Path
        Path to the generated XFOIL polar file.
    """

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_name = (
        f"naca{naca}_re{reynolds}_auto.txt"
    )

    output_path = (
        RAW_DATA_DIR / output_name
    )

    # Remove the previous output file so every
    # simulation starts with a clean result file.
    if output_path.exists():
        output_path.unlink()

    # -----------------------------------------------------
    # Commands sent directly to XFOIL
    # -----------------------------------------------------

    commands = [
        f"NACA {naca}",
        "OPER",

        f"VISC {reynolds}",

        # Give difficult cases more iterations
        "ITER 200",

        # Start saving polar data
        "PACC",
        output_name,

        # No dump file
        "",

        # Positive angle sweep:
        # 0, 1, 2, ... 12 degrees
        f"ASEQ 0 {alpha_end} {alpha_step}",

        # Reset boundary-layer solution before
        # changing sweep direction
        "INIT",

        # Negative angle sweep:
        # -1, -2, ... -6 degrees
        f"ASEQ -1 {alpha_start} {-alpha_step}",

        # Stop polar accumulation
        "PACC",

        # Return from OPER menu
        "",

        # Exit XFOIL
        "QUIT",
    ]

    command_text = (
        "\n".join(commands) + "\n"
    )

    print()
    print("--------------------------------------")
    print("Running XFOIL")
    print("--------------------------------------")
    print(f"Airfoil: NACA {naca}")
    print(f"Reynolds number: {reynolds}")
    print(
        f"Angle range: "
        f"{alpha_start} to {alpha_end} degrees"
    )

    try:
        result = subprocess.run(
            [str(XFOIL_EXE)],
            input=command_text,
            text=True,
            cwd=RAW_DATA_DIR,
            capture_output=True,

            # Allow difficult cases more time
            timeout=300,
        )

    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"XFOIL timed out for NACA {naca} "
            f"at Re = {reynolds}"
        ) from error

    if result.returncode != 0:
        print()
        print("XFOIL returned an error:")
        print(result.stderr)

    if not output_path.exists():
        raise FileNotFoundError(
            f"XFOIL finished but did not create "
            f"the polar file for NACA {naca}, "
            f"Re = {reynolds}."
        )

    print()
    print("XFOIL run completed successfully.")
    print("Results saved to:")
    print(output_path)

    return output_path


if __name__ == "__main__":
    run_xfoil()