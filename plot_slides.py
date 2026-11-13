from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.optimize import least_squares
from scipy.stats import linregress

root = Path(__file__).resolve().parent
out = root / 'results'
out.mkdir(exist_ok=True)
cal = pd.read_csv(root / 'data/fixed_bead/calibration.csv')
data = pd.read_csv(out / 'measurements.csv')
selected = pd.read_csv(out / 'alpha_selected.csv')

file = next((root / 'data/fixed_bead/1um/x').glob('*300.2*txt'))
raw = np.loadtxt(file, usecols=(0, 2))
q = raw[:, 0][:10000]
x = (raw[:, 1][:10000] - np.median(raw[:, 1][:10000])) * 2
imin = np.argmin(q)
part = q[max(0, imin - 500):min(len(q), imin + 501)]
imax = max(0, imin - 500) + np.argmax(part)
branch = np.arange(min(imin, imax), max(imin, imax) + 1)
lo, hi = np.min(q[branch]), np.max(q[branch])
fit = branch[(q[branch] >= lo + .125 * (hi - lo)) & (q[branch] <= hi - .125 * (hi - lo))]
m, b = np.polyfit(x[fit], q[fit], 1)
fig, ax = plt.subplots(figsize=(7.3, 5.1))
ax.plot(x, q, '.', ms=1.1, alpha=.35, label='All data')
ax.scatter(x[fit], q[fit], s=7, label='Central 75% fit region')
xx = np.linspace(x[fit].min(), x[fit].max(), 100)
ax.plot(xx, m * xx + b, lw=2, label='Linear fit')
ax.set_title('Example QPD calibration curve\n1.05 µm bead, X-axis scan, 300.2 mA')
ax.set_xlabel('Stage displacement (µm)')
ax.set_ylabel('QPD voltage (V)')
ax.legend()
fig.tight_layout()
fig.savefig(out / 'calibration_graph.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.85), sharey=True)
for ax, size, title in zip(axes, (1, 3), ('1.05 µm beads', '3.17 µm beads')):
    for axis, color, marker in [('X', 'tab:blue', 'o'), ('Y', 'tab:orange', 's')]:
        group = cal[(cal.bead_um == size) & (cal.axis == axis)].sort_values('current_mA')
        current = group.current_mA.to_numpy()
        slopes = np.abs(group.baseline_slope.to_numpy())
        fit = linregress(current, slopes)
        grid = np.linspace(100, 500, 200)
        ax.scatter(current, slopes, marker=marker, color=color, s=30, label=f'{axis} data')
        ax.plot(grid, fit.slope * grid + fit.intercept, color=color, label=f'{axis} fit ($R^2$={fit.rvalue ** 2:.3f})')
    ax.set_title(title)
    ax.set_xlabel('Laser current (mA)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc='upper left')
axes[0].set_ylabel('QPD calibration slope (V/V)')
fig.tight_layout()
fig.savefig(out / 'qpd_calibration.png', dpi=150)
plt.close(fig)

row = data[(data.size_folder == 1) & (data.axis == 'X') & (np.isclose(data.current_mA, 300.4))].iloc[0]
file = next((root / 'data/floating_bead/1um').glob('*300.4*txt'))
signal = np.loadtxt(file, usecols=(0,))
f, p = welch(signal - signal.mean(), fs=5000, window='hann', nperseg=8192, noverlap=4096)
edges = np.geomspace(f[1], f[-1], 101)
which = np.digitize(f, edges)
bins = [(f[which == j].mean(), p[which == j].mean()) for j in range(1, len(edges)) if np.any(which == j)]
fb, pb = np.asarray(bins).T
fit_range = (fb >= 5) & (fb <= 1000)
frequency = fb[fit_range]
observed = pb[fit_range]

def errors(z):
    A, fc, noise = np.exp(z)
    return np.log(A / (frequency**2 + fc**2) + noise) - np.log(observed)

choices = []
for guess in (5, 20, 70, 200, 600):
    start = [max(observed[0] * (frequency[0]**2 + guess**2), 1e-14), guess, max(np.median(observed[-8:]) * .5, 1e-15)]
    choices.append(least_squares(errors, np.log(start), bounds=(np.log([1e-20, .2, 1e-17]), np.log([1e8, 3000, 1e3])), max_nfev=350))
A, fc, noise = np.exp(min(choices, key=lambda item: np.sum(item.fun**2)).x)
scale = row.displacement_m_per_V
fig, ax = plt.subplots(figsize=(7.3, 4.55))
ax.loglog(frequency, observed * scale**2, '.', markersize=4, label='Measured PSD')
grid = np.geomspace(5, 1000, 400)
ax.loglog(grid, (A / (grid**2 + fc**2) + noise) * scale**2, color='tab:orange', lw=2, label='Lorentzian fit')
ax.set_title('PSD Fit — 1.05 µm Bead, X-axis, 300 mA')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (m²/Hz)')
ax.legend()
fig.tight_layout()
fig.savefig(out / 'Lorentizan.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True)
for ax, axis in zip(axes, ['X', 'Y']):
    grid = np.linspace(100, 500, 300)
    fits = []
    for bead, color, marker in [(1.05, 'tab:blue', 'o'), (3.17, 'tab:orange', 's')]:
        sub = selected[np.isclose(selected.bead_um, bead) & (selected.axis == axis)].sort_values('current_mA')
        current = sub.current_mA.to_numpy()
        stiffness = sub.alpha_pN_um.to_numpy()
        fit = linregress(current, stiffness)
        fits.append(fit)
        ax.scatter(current, stiffness, color=color, marker=marker, s=43, label=f'{bead:.2f} µm data', zorder=3)
        ax.plot(grid, fit.slope * grid + fit.intercept, color=color, ls='--', lw=1.8, label=f'{bead:.2f} µm fit')
    m = np.mean([fit.slope for fit in fits])
    b = np.mean([fit.intercept for fit in fits])
    ax.plot(grid, m * grid + b, color='black', lw=2.6, label=f'Average: α = {m:.5f}I {b:+.4f}')
    ax.set_title(f'{axis}-axis')
    ax.set_xlabel('Laser supply current, I (mA)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc='upper left')
axes[0].set_ylabel('Trap stiffness, α (pN/µm)')
fig.suptitle('Trap stiffness vs. laser current — all measurements included')
fig.tight_layout()
fig.savefig(out / 'trap_stiffness.png', dpi=230, bbox_inches='tight')
plt.close(fig)

accepted = 1.380649e-23
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
all_values = np.concatenate([data.kB_PSD_J_K.to_numpy(), data.kB_equip_J_K.to_numpy()])
bins = np.geomspace(all_values.min() * .8, all_values.max() * 1.2, 24)
for ax, title, column in zip(axes, ['PSD method', 'Equipartition method'], ['kB_PSD_J_K', 'kB_equip_J_K']):
    values = data[column].to_numpy()
    ax.hist(values, bins=bins, color='#4c78a8', edgecolor='white')
    ax.axvline(accepted, color='tab:orange', ls='--', lw=2, label='Accepted $k_B$')
    ax.axvline(np.median(values), color='tab:green', lw=2, label='Median')
    ax.set_xscale('log')
    ticks = accepted * 10.0**np.arange(-1, 4)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{tick:.2e}' for tick in ticks], fontsize=8)
    ax.set_title(title)
    ax.set_xlabel('Estimated kB (J/K)')
    ax.grid(axis='y', alpha=.2)
    ax.legend(fontsize=9)
axes[0].set_ylabel('Number of measurements')
fig.tight_layout()
fig.savefig(out / 'kb_histograms.png', dpi=150)
plt.close(fig)
