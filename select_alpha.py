from pathlib import Path
import numpy as np
import pandas as pd
root = Path(__file__).resolve().parent
out = root / 'results'
df = pd.read_csv(out / 'measurements.csv')
repeat = np.isclose(df.bead_um, 1.05) & df.current_mA.between(295, 305)
chosen = repeat & np.isclose(df.current_mA, 301.5, atol=0.1)
assert chosen.sum() == 2, 'Expected X and Y of 301.5 mA run'
selected = df[~repeat | chosen].copy()
selected.to_csv(out / 'alpha_selected.csv', index=False)
print('Stiffness plot uses', len(selected), 'axes from', selected.file.nunique(), 'recordings')
