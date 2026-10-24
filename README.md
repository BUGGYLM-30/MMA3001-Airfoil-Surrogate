# MMA3001-Airfoil-Surrogate
Machine learning surrogate model for predicting NACA airfoil aerodynamic coefficients

# MMA3001 Airfoil Surrogate Model

## Project Overview

This project develops a machine-learning surrogate model for predicting the aerodynamic performance of NACA 4-digit airfoils.

The surrogate predicts:

- coefficient of lift (`CL`)
- coefficient of drag (`CD`)

using five aerodynamic and geometric inputs:

- maximum camber
- position of maximum camber
- airfoil thickness
- Reynolds number
- angle of attack

The primary aerodynamic dataset was generated using XFOIL and Python automation. Linear and polynomial regression models were then developed and compared.

The final selected model was a degree-3 polynomial regression surrogate.

---

## Engineering Motivation

XFOIL can provide useful aerodynamic predictions for two-dimensional airfoils, but repeated numerical simulations become computationally expensive when many airfoil geometries and operating conditions must be evaluated.

A surrogate model provides a faster approximation by learning the relationship between airfoil geometry, operating conditions and aerodynamic coefficients.

The intended application is rapid preliminary aerodynamic analysis within the defined model domain.

The surrogate is not intended to replace high-fidelity CFD or experimental wind-tunnel testing.

---

## Model Inputs

The surrogate accepts the following inputs:

| Input | Description | Domain |
|---|---|---|
| Maximum camber | Maximum airfoil camber as a chord fraction | 0.00–0.04 |
| Camber position | Position of maximum camber as a chord fraction | 0.20–0.60 for cambered airfoils |
| Thickness | Maximum thickness as a chord fraction | 0.09–0.15 |
| Reynolds number | Dimensionless Reynolds number | 100,000–500,000 |
| Angle of attack | Airfoil angle relative to the freestream | −6° to 12° |

For symmetric airfoils, maximum camber and camber position are both represented as zero.

---

## Model Outputs

The model predicts:

- `CL` — coefficient of lift
- `CD` — coefficient of drag

Both outputs are dimensionless.

---

## Dataset

The final aerodynamic dataset contains 1,925 usable operating points generated with XFOIL.

The requested dataset contained 1,995 operating points, resulting in approximately 96.5% overall coverage.

The final dataset contains:

- 21 NACA 4-digit airfoils
- 5 Reynolds numbers
- angles of attack from −6° to 12°

Most missing data resulted from individual XFOIL convergence failures.

One complete XFOIL case, NACA 0009 at Reynolds number 200,000, failed after repeated attempts and was retained as a documented numerical limitation rather than replaced with fabricated data.

---

## Computational Workflow

The main workflow is:

1. Define NACA 4-digit airfoil geometries.
2. Automate XFOIL using Python.
3. Generate aerodynamic polar data.
4. Parse and combine XFOIL results.
5. Audit convergence and dataset quality.
6. Develop a multiple linear regression baseline.
7. Develop degree-2 and degree-3 polynomial regression models.
8. Validate models using withheld airfoil geometries.
9. Perform grouped cross-validation.
10. Analyse model errors and input sensitivity.
11. Compare against independent wind-tunnel data.
12. Compare surrogate prediction runtime against XFOIL.

---

## Model Development

Three regression approaches were compared:

- multiple linear regression
- degree-2 polynomial regression
- degree-3 polynomial regression

Whole airfoil geometries were withheld during validation rather than randomly splitting individual operating points. This reduces information leakage and tests whether the surrogate can generalise to previously unseen geometries.

Five-fold grouped cross-validation was also performed using airfoil identity as the grouping variable.

### Mean Grouped Cross-Validation Results

| Model | Lift MAE | Lift R² | Drag MAE | Drag R² |
|---|---:|---:|---:|---:|
| Linear regression | 0.07284 | 0.97193 | 0.00693 | 0.26997 |
| Degree-2 polynomial | 0.04582 | 0.98802 | 0.00489 | 0.62366 |
| Degree-3 polynomial | 0.04216 | 0.98930 | 0.00480 | 0.67345 |

Degree-3 polynomial regression was selected because it produced the best average performance across the grouped validation folds.

---

## Independent Experimental Validation

Independent experimental lift data for the NACA 0009 airfoil were obtained from the UIUC/Princeton low-speed airfoil wind-tunnel dataset.

NACA 0009 was removed entirely from the surrogate training dataset for this validation.

At approximately Reynolds number 100,000:

| Method | MAE | RMSE | R² |
|---|---:|---:|---:|
| XFOIL vs wind tunnel | 0.06498 | 0.07141 | 0.97289 |
| Surrogate vs wind tunnel | 0.08558 | 0.10231 | 0.94436 |

The surrogate therefore reproduced the experimental lift trend reasonably well for an unseen airfoil, although XFOIL remained more accurate relative to the experimental measurements.

This experimental comparison validates lift only and should not be interpreted as independent experimental validation of drag.

---

## Sensitivity and Error Analysis

The model was analysed with respect to:

- angle of attack
- Reynolds number
- maximum camber
- camber position
- thickness

Prediction errors generally increased at low Reynolds numbers and near the extremes of the angle-of-attack range.

The lowest Reynolds number, 100,000, produced the largest average lift and drag errors.

Angle of attack had a strong influence on lift, while drag exhibited a considerably more nonlinear response.

The surrogate should therefore be used only inside its defined input domain and with additional caution at low Reynolds numbers and extreme angles of attack.

---

## Computational Performance

The final surrogate was benchmarked using 10,000 operating-point predictions.

Measured performance was approximately:

- 10,000 predictions in 0.0129 seconds
- approximately 774,000 operating-point predictions per second
- approximately 76,000× estimated speed-up relative to XFOIL per operating point

The speed-up figure represents a batch prediction comparison and should be interpreted as an approximate computational performance comparison rather than a universal execution-time ratio.

---

## Repository Structure

```text
MMA3001-Airfoil-Surrogate/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 00_setup_test.ipynb
│   ├── 01_xfoil_proof.ipynb
│   ├── 02_verify_xfoil_automation.ipynb
│   ├── 03_pilot_dataset_analysis.ipynb
│   ├── 04_final_dataset_audit.ipynb
│   ├── 05_baseline_model.ipynb
│   ├── 06_model_analysis.ipynb
│   ├── 07_experimental_validation.ipynb
│   └── 08_performance_analysis.ipynb
│
├── results/
│
├── src/
│   ├── generate_dataset.py
│   └── xfoil_runner.py
│
├── tests/
│   └── test_project.py
│
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Python Environment

The project was developed using Python 3.12.

Install the required Python packages using:

```powershell
python -m pip install -r requirements.txt
```

Main libraries include:

- NumPy
- pandas
- matplotlib
- scikit-learn
- pytest
- Jupyter

XFOIL must also be installed separately.

The XFOIL executable location in `src/xfoil_runner.py` may need to be updated for another computer.

---

## Running the Dataset Generator

From the project root:

```powershell
python src/generate_dataset.py
```

This generates the aerodynamic dataset and convergence summary in:

```text
data/processed/
```

Because XFOIL is an external numerical solver, convergence is not guaranteed for every requested operating point.

---

## Running the Tests

From the project root:

```powershell
python -m pytest -v
```

The current automated test suite contains eight tests covering:

- NACA geometry parsing
- symmetric-airfoil handling
- invalid NACA input handling
- final dataset availability
- required dataset columns
- missing model values
- duplicate rows
- model input-domain limits

The current test suite passes all eight tests.

A test report is stored in:

```text
results/test_report.txt
```

---

## Limitations

Important limitations include:

- only NACA 4-digit airfoils are represented;
- the model is limited to two-dimensional aerodynamic behaviour;
- the training data are primarily generated by XFOIL;
- XFOIL itself contains modelling assumptions and numerical limitations;
- some requested XFOIL operating points did not converge;
- the surrogate should not be extrapolated beyond its training domain;
- prediction accuracy decreases at some low-Reynolds-number and high-angle-of-attack conditions;
- independent experimental validation was performed for lift using one airfoil and does not establish universal experimental accuracy;
- the surrogate is intended for preliminary analysis rather than replacing CFD or wind-tunnel testing.

---

## Testing and Reproducibility

The project uses:

- GitHub version control
- reproducible Python scripts
- Jupyter notebooks
- automated pytest tests
- documented dataset generation
- independent validation
- explicit input-domain restrictions

The repository commit history records the development of the data-generation, modelling, validation and testing workflow.

---

## AI Use

AI tools were used during project development for assistance with planning, Python code development, debugging, interpretation of results and documentation.

All generated code and technical suggestions were executed, checked and evaluated during the project.

Important engineering decisions, model selection, validation interpretation and submitted conclusions remain the responsibility of the student.

A more detailed AI-use reflection is provided in the final project documentation.

---

## Author

Levi McFarland  
MMA3001 — Numerical Methods and Machine Learning  
Monash University