from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from core import SEED, fit_linear, standardize
from experiments.common import plt, save_figure, show_result


PLOT = "task_7_linear.png"


def run(make_plot=False):
    # Fit petal width from petal length and measure error on untouched test flowers.
    iris = load_iris()
    x = iris.data[:, [2]]
    y = iris.data[:, 3]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
    train, test, _, _ = standardize(x_train, x_test)
    weights, history = fit_linear(train, y_train)
    predictions = np.column_stack([np.ones(len(test)), test]) @ weights
    result = {"intercept_scaled": float(weights[0]), "slope_scaled": float(weights[1]),
              "test_mse": float(mean_squared_error(y_test, predictions)),
              "initial_loss": history[0], "final_loss": history[-1]}
    if make_plot:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].scatter(x_test[:, 0], y_test, label="Held-out measurements")
        order = np.argsort(x_test[:, 0])
        axes[0].plot(x_test[order, 0], predictions[order], color="crimson", label="Manual fit")
        axes[0].set(xlabel="Petal length (cm)", ylabel="Petal width (cm)")
        axes[0].legend()
        axes[1].plot(history)
        axes[1].set(xlabel="Gradient step", ylabel="Training MSE", title="Gradient descent curve")
        save_figure(PLOT)
    return result


if __name__ == "__main__":
    show_result(7, run(), PLOT)
