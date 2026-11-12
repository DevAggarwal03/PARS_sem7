from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import first_pca_axis
from experiments.common import show_result
from experiments.projection import finish_projection, projection_data


PLOT = "task_10_pca.png"


def run(make_plot=False):
    # Select the greatest variance direction and measure classification after one dimensional PCA projection.
    data = projection_data()
    center, axis = first_pca_axis(data[1])
    return finish_projection(data, center, axis, "PCA", PLOT, make_plot)


if __name__ == "__main__":
    show_result(10, run(), PLOT)
