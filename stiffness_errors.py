from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.optimize import least_squares
from scipy.stats import linregress

root = Path(__file__).resolve().parent
out = root / 'results'
measurements = pd.read_csv(out / 'measurements.csv')
selected = pd.read_csv(out / 'alpha_selected.csv')


def fit_corner(signal):
    f, p = welch(signal - np.mean(signal), fs=5000, window='hann', nperseg=8192, noverlap=4096)
    edges = np.geomspace(f[1], f[-1], 101)
    groups = np.digitize(f, edges)
    bins = [(f[groups == j].mean(), p[groups == j].mean())
            for j in range(1, len(edges)) if np.any(groups == j)]
    fb, pb = np.array(bins).T
    keep = (fb >= 5) & (fb <= 1000)
    x, y = fb[keep], pb[keep]

    def residual(z):
        A, fc, noise = np.exp(z)
        return np.log(A / (x**2 + fc**2) + noise) - np.log(y)

    fits = []
    for guess in [5, 20, 70, 200, 600]:
        start = [max(y[0] * (x[0]**2 + guess**2), 1e-14),
                 guess, max(np.median(y[-8:]) * 0.5, 1e-15)]
        fit = least_squares(residual, np.log(start),
                            bounds=(np.log([1e-20, .2, 1e-17]),
                                    np.log([1e8, 3000, 1e3])), max_nfev=350)
        fits.append(fit)
    return np.exp(min(fits, key=lambda fit: np.sum(fit.fun**2)).x[1])


rows = []
for file in measurements.file.unique():
    info = measurements[measurements.file == file]
    size = int(info.size_folder.iloc[0])
    signal = np.loadtxt(root / 'data/floating_bead' / f'{size}um' / file, usecols=(0, 1))
    for axis, column in [('X', 0), ('Y', 1)]:
        corners = [fit_corner(piece) for piece in np.array_split(signal[:, column], 5)]
        row = info[info.axis == axis].iloc[0]
        sigma_fc = np.std(corners, ddof=1) / np.sqrt(5)
        alpha = row.alpha_pN_um
        sigma_fit = alpha * sigma_fc / row.corner_frequency_Hz
        sigma_eta = alpha * 0.015
        sigma_d = alpha * 0.005 / row.bead_um
        rows.append(dict(file=file, axis=axis, bead_um=row.bead_um,
                         current_mA=row.current_mA, f0_Hz=row.corner_frequency_Hz,
                         sigma_f0_Hz=sigma_fc, alpha_pN_um=alpha,
                         sigma_alpha_fit=sigma_fit, sigma_alpha_eta=sigma_eta,
                         sigma_alpha_diameter=sigma_d,
                         sigma_alpha_total=np.sqrt(sigma_fit**2 + sigma_eta**2 + sigma_d**2)))

errors = pd.DataFrame(rows)
errors.to_csv(out / 'stiffness_point_errors.csv', index=False)
points = errors.merge(selected[['file', 'axis']], on=['file', 'axis'])
results = []
for axis in ['X', 'Y']:
    for bead in [1.05, 3.17]:
        group = points[(points.axis == axis) & np.isclose(points.bead_um, bead)]
        x = group.current_mA.to_numpy()
        y = group.alpha_pN_um.to_numpy()
        fit = linregress(x, y)
        r = y - (fit.slope * x + fit.intercept)
        fit_error = group.sigma_alpha_fit.to_numpy()
        eta_error = group.sigma_alpha_eta.to_numpy()
        d_error = group.sigma_alpha_diameter.to_numpy()
        covariance = np.diag(fit_error**2) + np.outer(eta_error, eta_error) + np.outer(d_error, d_error)
        chi2 = float(r @ np.linalg.solve(covariance, r))
        results.append(dict(axis=axis, bead_um=bead, chi2=chi2, dof=len(x)-2))
    pair = [r for r in results if r['axis'] == axis]
    results.append(dict(axis=axis, bead_um='combined',
                        chi2=sum(r['chi2'] for r in pair), dof=sum(r['dof'] for r in pair)))

pd.DataFrame(results).to_csv(out / 'stiffness_chi2.csv', index=False)
print(pd.DataFrame(results).to_string(index=False))
