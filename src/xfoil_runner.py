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

    Parameters
    ----------
    naca : str
        Four-digit NACA airfoil code.
    reynolds : int
        Reynolds number.
    alpha_start : float
        Starting angle of attack in degrees.
    alpha_end : float
        Final angle of attack in degrees.
    alpha_step : float
        Angle-of-attack increment in degrees.

    Returns
    -------
    Path
        Path to the generated XFOIL polar file.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_name = f"naca{naca}_re{reynolds}_auto.txt"
    output_path = RAW_DATA_DIR / output_name

    # Remove old output so each run starts cleanly
    if output_path.exists():
        output_path.unlink()

    commands = [
        f"NACA {naca}",
        "OPER",
        f"VISC {reynolds}",
        "ITER 100",
        "PACC",
        output_name,
        "",  # No dump file
        f"ASEQ {alpha_start} {alpha_end} {alpha_step}",
        "PACC",
        "",  # Return from OPER menu
        "QUIT",
    ]

    command_text = "\n".join(commands) + "\n"

    print("Running XFOIL...")
    print(f"Airfoil: NACA {naca}")
    print(f"Reynolds number: {reynolds}")
    print(
        f"Angle range: {alpha_start} to "
        f"{alpha_end} degrees"
    )

    result = subprocess.run(
        [str(XFOIL_EXE)],
        input=command_text,
        text=True,
        cwd=RAW_DATA_DIR,
        capture_output=True,
        timeout=120,
    )

    if result.returncode != 0:
        print("XFOIL returned an error:")
        print(result.stderr)

    if not output_path.exists():
        raise FileNotFoundError(
            "XFOIL finished but the polar file was not created."
        )

    print()
    print("XFOIL run completed successfully.")
    print(f"Results saved to:")
    print(output_path)

    return output_path


if __name__ == "__main__":
    run_xfoil()