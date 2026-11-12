from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import first_lda_axis
from experiments.common import show_result
from experiments.projection import finish_projection, projection_data


PLOT = "task_11_lda.png"


def run(make_plot=False):
    # Select the class separating direction and measure classification after one dimensional LDA projection.
    data = projection_data()
    center, axis = first_lda_axis(data[1], data[3])
    return finish_projection(data, center, axis, "LDA", PLOT, make_plot)


if __name__ == "__main__":
    show_result(11, run(), PLOT)
