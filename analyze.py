from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.optimize import least_squares
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
FS = 5000
def water_viscosity(temp_c):
    return 2.414e-5 * 10 ** (247.8 / (temp_c + 133.15))

temperature_files = sorted((ROOT / 'data/floating_bead').rglob('*.txt'))
temperatures_f = np.array([float(re.search(r'_ltemp_([0-9.,]+)_stemp', f.name).group(1).replace(',', '.')) for f in temperature_files])
TEMP_MEAN_F = temperatures_f.mean()
TEMP_SD_F = temperatures_f.std(ddof=1)
TEMP_MEAN_C = (TEMP_MEAN_F - 32) * 5 / 9
T = TEMP_MEAN_C + 273.15
ETA = water_viscosity(TEMP_MEAN_C)
ETA_SIGMA = (water_viscosity(TEMP_MEAN_C - TEMP_SD_F * 5 / 9) - water_viscosity(TEMP_MEAN_C + TEMP_SD_F * 5 / 9)) / (2 * ETA)
print(f'Temperature: n={len(temperatures_f)}, mean={TEMP_MEAN_F:.4f} F, SD={TEMP_SD_F:.4f} F, range={temperatures_f.min():.1f}-{temperatures_f.max():.1f} F')
print(f'Water viscosity at mean temperature: {ETA:.6g} Pa s; relative sigma from +-1 SD: {ETA_SIGMA*100:.3f}%')
STAGE = 2e-6
KB = 1.380649e-23
DIAMETERS = {1: 1.05e-6, 3: 3.17e-6}
RNG = np.random.default_rng(20261107)
cal = pd.read_csv(ROOT / 'data/fixed_bead/calibration.csv')
models = {}
for (size, axis), group in cal.groupby(['bead_um', 'axis']):
    group = group.sort_values('current_mA')
    coeff = np.polyfit(group.current_mA, group.mc_mean_slope, 1)
    models[size, axis] = (coeff, group)

def fit_psd(q):
    f, p = welch(q - np.mean(q), fs=FS, window='hann', nperseg=8192, noverlap=4096)
    edges = np.geomspace(f[1], f[-1], 101)
    groups = np.digitize(f, edges)
    bins = []
    for j in range(1, len(edges)):
        points = groups == j
        if np.any(points):
            bins.append((f[points].mean(), p[points].mean()))
    fb, pb = np.array(bins).T
    mask = (fb >= 5) & (fb <= 1000)
    x = fb[mask]
    y = pb[mask]

    def residual(z):
        A, fc, noise = np.exp(z)
        return np.log(A / (x * x + fc * fc) + noise) - np.log(y)
    fits = []
    for guess in (5, 20, 70, 200, 600):
        initial = [
            max(y[0] * (x[0] ** 2 + guess ** 2), 1e-14),
            guess,
            max(np.median(y[-8:]) * 0.5, 1e-15)
        ]
        fit = least_squares(
            residual,
            np.log(initial),
            bounds=(np.log([1e-20, 0.2, 1e-17]), np.log([1e8, 3000, 1e3])),
            max_nfev=350
        )
        fits.append(fit)
    A, fc, noise = np.exp(min(fits, key=lambda r: np.sum(r.fun ** 2)).x)
    return A, fc, noise, fb, pb
rows = []
for file in sorted((ROOT / 'data/floating_bead').rglob('*.txt')):
    size = int(file.parent.name.replace('um', ''))
    current = float(re.search('_([\\d.]+)_curr(?:ent|emt)', file.name).group(1))
    signals = np.loadtxt(file, usecols=(0, 1))
    for axis, column in [('X', 0), ('Y', 1)]:
        coeff, _ = models[size, axis]
        slope = abs(np.polyval(coeff, current))
        scale = STAGE / slope
        q = signals[:, column]
        A_volts, fc, noise, fb, pb = fit_psd(q)
        beta = 3 * np.pi * ETA * DIAMETERS[size]
        alpha = 2 * np.pi * beta * fc
        variance = np.var(q * scale, ddof=1)
        rows.append(dict(
            psd_window='hann',
            welch_segment_points=8192,
            welch_overlap_points=4096,
            log_frequency_bins_requested=100,
            fit_frequency_min_Hz=5,
            fit_frequency_max_Hz=1000,
            fitted_bin_count=int(np.sum((fb >= 5) & (fb <= 1000))),
            file=file.name,
            bead_um=DIAMETERS[size] * 1e6,
            size_folder=size,
            axis=axis,
            current_mA=current,
            calibration_slope_V_V=slope,
            displacement_m_per_V=scale,
            eta_Pa_s=ETA,
            temperature_K=T,
            beta_N_s_m=beta,
            corner_frequency_Hz=fc,
            alpha_N_m=alpha,
            alpha_pN_um=alpha * 1e6,
            variance_m2=variance,
            A_m2_Hz=A_volts * scale ** 2,
            kB_PSD_J_K=np.pi ** 2 * beta * A_volts * scale ** 2 / T,
            kB_equip_J_K=alpha * variance / T
        ))

data = pd.DataFrame(rows).sort_values(['size_folder', 'current_mA', 'axis'])
data.to_csv(OUT / 'measurements.csv', index=False)
pd.DataFrame([dict(n_recordings=len(temperatures_f), mean_F=TEMP_MEAN_F, sd_F=TEMP_SD_F, min_F=temperatures_f.min(), max_F=temperatures_f.max(), mean_K=T, eta_Pa_s=ETA, eta_rel_sigma=ETA_SIGMA)]).to_csv(OUT / 'temperature_viscosity.csv', index=False)
files = data.file.unique()
arr = data.reset_index(drop=True)
record_groups = [arr.index[arr.file == f].to_numpy() for f in files]
assert len(files) == 20 and all(len(g) == 2 for g in record_groups)
N = 10000
draws = RNG.integers(len(files), size=(N, len(files)))
idx = np.array([
    np.concatenate([record_groups[i] for i in draw])
    for draw in draws
])
eta_factor = 1 + RNG.normal(0, ETA_SIGMA, size=N)
correction = np.ones((N, len(arr))) * eta_factor[:, None]
for size in (1, 3):
    d_factor = (DIAMETERS[size] + RNG.normal(0, 0.005e-6, N)) / DIAMETERS[size]
    for axis in ('X', 'Y'):
        coeff, group = models[size, axis]
        x = group.current_mA.to_numpy()
        y = group.mc_mean_slope.to_numpy()
        noise = RNG.normal(size=(N, len(x))) * group.mc_sd_slope.to_numpy()
        sample_slopes = y + noise
        slopes = (sample_slopes - sample_slopes.mean(axis=1)[:, None]) @ (x - x.mean())
        slopes = slopes / np.sum((x - x.mean()) ** 2)
        intercepts = sample_slopes.mean(axis=1) - slopes * x.mean()
        mask = (arr.size_folder == size) & (arr.axis == axis)
        current = arr.loc[mask, 'current_mA'].to_numpy()
        original = arr.loc[mask, 'calibration_slope_V_V'].to_numpy()
        new = np.abs(slopes[:, None] * current[None, :] + intercepts[:, None])
        correction[:, mask.to_numpy()] *= d_factor[:, None] * (original[None, :] / new) ** 2
values = []
for key in ('kB_PSD_J_K', 'kB_equip_J_K'):
    adjusted = arr[key].to_numpy()[None, :] * correction
    medians = np.median(adjusted[np.arange(N)[:, None], idx], axis=1)
    values.append(medians)
values = np.column_stack(values)
summary = []
for method, key, col in [('PSD', 'kB_PSD_J_K', 0), ('Equipartition', 'kB_equip_J_K', 1)]:
    v = arr[key].to_numpy()
    quantiles = np.percentile(values[:, col], [0.15, 2.5, 16, 84, 97.5, 99.85])
    summary.append(dict(
        method=method,
        median_J_K=np.median(v),
        CI68_low=quantiles[2],
        CI68_high=quantiles[3],
        CI95_low=quantiles[1],
        CI95_high=quantiles[4],
        CI997_low=quantiles[0],
        CI997_high=quantiles[5]
    ))
pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
print(f'{len(files)} recordings, {len(arr)} axis measurements')
print(pd.DataFrame(summary).to_string(index=False))
print('\nAlpha at ~300 mA (N/m and pN/µm):')
print(arr[(arr.current_mA - 300).abs() < 5][['bead_um', 'axis', 'current_mA', 'corner_frequency_Hz', 'alpha_N_m', 'alpha_pN_um']].to_string(index=False))
