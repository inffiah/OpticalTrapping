import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
for script in ['analyze.py', 'select_alpha.py', 'fit_alpha_all_points_uncertainty.py', 'plot_slides.py']:
    subprocess.run([sys.executable, str(root / script)], cwd=root, check=True)
