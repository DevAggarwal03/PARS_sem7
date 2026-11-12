import json
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import PLOTS, ensure_dirs


def save_figure(name):
    # Save a readable experiment figure and release its plotting canvas after writing the image.
    ensure_dirs()
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=180, bbox_inches="tight")
    plt.close()


def show_result(number, result, plot_name):
    # Print fresh measurements and open the existing figure without changing saved assignment files.
    path = PLOTS / plot_name
    if not path.is_file():
        raise FileNotFoundError(f"Saved plot is missing: {path}")
    print(f"Practical {number} results")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Opening saved plot: {path}", flush=True)
    subprocess.run(["open", str(path)], check=True)
