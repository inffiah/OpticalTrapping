from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

root = Path(__file__).resolve().parent
out = root / 'results'
cal = pd.read_csv(root / 'data/fixed_bead/calibration.csv')
data = pd.read_csv(out / 'measurements.csv')
selected = pd.read_csv(out / 'alpha_selected.csv')
errors = pd.read_csv(out / 'stiffness_point_errors.csv')
points = selected.merge(errors[['file', 'axis', 'sigma_alpha_total']], on=['file', 'axis'])

fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.85), sharey=True)
for ax, size, title in zip(axes, [1, 3], ['1.05 µm beads', '3.17 µm beads']):
    for axis, color, marker in [('X', 'tab:blue', 'o'), ('Y', 'tab:orange', 's')]:
        group = cal[(cal.bead_um == size) & (cal.axis == axis)].sort_values('current_mA')
        current = group.current_mA.to_numpy()
        slope = np.abs(group.baseline_slope.to_numpy())
        sigma = group.mc_sd_slope.to_numpy()
        fit = linregress(current, slope)
        x = np.linspace(100, 500, 200)
        ax.plot(x, fit.slope*x + fit.intercept, color=color,
                label=f'{axis} fit ($R^2$={fit.rvalue**2:.3f})')
        ax.errorbar(current, slope, xerr=0.05, yerr=sigma, color=color,
                    fmt=marker, ms=4.5, capsize=2.5, linestyle='none',
                    label=f'{axis} data (1σ)')
    ax.set_title(title)
    ax.set_xlabel('Laser current (mA)')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc='upper left')
axes[0].set_ylabel('QPD calibration slope (V/V)')
fig.tight_layout()
fig.savefig(out / 'qpd_calibration.png', dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
for ax, axis in zip(axes, ['X', 'Y']):
    xfit = np.linspace(100, 500, 200)
    fits = []
    for bead, color, marker in [(1.05, 'tab:blue', 'o'), (3.17, 'tab:orange', 's')]:
        group = points[(points.axis == axis) & np.isclose(points.bead_um, bead)].sort_values('current_mA')
        x = group.current_mA.to_numpy()
        y = group.alpha_pN_um.to_numpy()
        fit = linregress(x, y)
        fits.append(fit)
        ax.errorbar(x, y, yerr=group.sigma_alpha_total.to_numpy(),
                    fmt=marker, color=color, markersize=5, capsize=3,
                    linestyle='none', label=f'{bead:.2f} µm data')
        ax.plot(xfit, fit.slope*xfit + fit.intercept, '--', color=color,
                label=f'{bead:.2f} µm fit')
    slope = np.mean([fit.slope for fit in fits])
    intercept = np.mean([fit.intercept for fit in fits])
    ax.plot(xfit, slope*xfit + intercept, color='black', lw=2, label='Average fit')
    ax.set_title(f'{axis}-axis')
    ax.set_xlabel('Laser current (mA)')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
axes[0].set_ylabel('Trap stiffness (pN/µm)')
fig.suptitle('Trap stiffness with uncertainties from five recording portions')
fig.tight_layout()
fig.savefig(out / 'trap_stiffness.png', dpi=200)
plt.close(fig)

accepted = 1.380649e-23
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
all_values = np.concatenate([data.kB_PSD_J_K.to_numpy(), data.kB_equip_J_K.to_numpy()])
bins = np.geomspace(all_values.min()*0.8, all_values.max()*1.2, 24)
for ax, title, column in zip(axes, ['PSD method', 'Equipartition method'],
                              ['kB_PSD_J_K', 'kB_equip_J_K']):
    values = data[column].to_numpy()
    ax.hist(values, bins=bins, color='#4c78a8', edgecolor='white')
    ax.axvline(accepted, color='tab:orange', ls='--', lw=2, label='Accepted $k_B$')
    ax.axvline(np.median(values), color='tab:green', lw=2, label='Median')
    ax.set_xscale('log')
    ticks = accepted * 10**np.arange(-1, 4, dtype=float)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{tick:.2e}' for tick in ticks], fontsize=8)
    ax.set_title(title)
    ax.set_xlabel('Estimated kB (J/K)')
    ax.grid(axis='y', alpha=0.2)
    ax.legend(fontsize=9)
axes[0].set_ylabel('Number of measurements')
fig.tight_layout()
fig.savefig(out / 'kb_histograms.png', dpi=200)
plt.close(fig)
