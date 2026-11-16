# Optical trapping analysis (temperature updated)

Open this folder in VS Code, install dependencies, then run:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

The program uses all 20 floating-bead temperature readings (mean 80.555 F, sample SD 2.701 F). It uses the viscosity at the mean temperature and a viscosity uncertainty calculated from the mean +/- one sample SD. The output plots and CSVs are written to `results/`.

Figures:
- `results/qpd_calibration.png` (unchanged by the temperature update)
- `results/trap_stiffness.png` (temperature-updated)
- `results/kb_histograms.png` (temperature-updated)

The numerical fit procedures remain unweighted; the plotted stiffness error bars and chi-square evaluation use propagated uncertainties.
