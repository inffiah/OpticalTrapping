from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import linregress
root = Path(__file__).resolve().parent
out = root / 'results'
out.mkdir(exist_ok=True)
df = pd.read_csv(out / 'alpha_selected.csv')
rows = []
fits = {}
for axis in ('X', 'Y'):
    for bead, color, marker in [(1.05, 'tab:blue', 'o'), (3.17, 'tab:orange', 's')]:
        sub = df[np.isclose(df.bead_um, bead) & (df.axis == axis)].sort_values('current_mA')
        x = sub.current_mA.to_numpy()
        y = sub.alpha_pN_um.to_numpy()
        r = linregress(x, y)
        fits[bead, axis] = r
        rows.append(dict(bead_um=bead, axis=axis, n=len(x), slope=r.slope, slope_SE=r.stderr, intercept=r.intercept, intercept_SE=r.intercept_stderr, R2=r.rvalue ** 2))
    rx, ry = [fits[bead, axis] for bead in (1.05, 3.17)]
    m = (rx.slope + ry.slope) / 2
    b = (rx.intercept + ry.intercept) / 2
    sm = np.hypot(rx.stderr, ry.stderr) / 2
    sb = np.hypot(rx.intercept_stderr, ry.intercept_stderr) / 2
    rows.append(dict(bead_um='equal-weight average', axis=axis, n=18, slope=m, slope_SE=sm, intercept=b, intercept_SE=sb, R2=np.nan))
pd.DataFrame(rows).to_csv(out / 'alpha_all_points_fit_uncertainties.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
