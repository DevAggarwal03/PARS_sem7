from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from core import SEED, fit_binary, polynomial_features, sigmoid, standardize
from experiments.common import plt, save_figure, show_result


PLOT = "task_9_regularization.png"


def run(make_plot=False):
    # Compare two logistic models on identical flowers with and without the L2 penalty.
    ds = load_iris()
    subset = ds.target > 0
    x = ds.data[subset][:, [2, 3]]
    y = (ds.target[subset] == 2).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.55, stratify=y, random_state=SEED
    )
    raw_train = polynomial_features(x_train)
    raw_test = polynomial_features(x_test)
    train, test, mean, scale = standardize(raw_train, raw_test)
    models = {}
    for label, penalty in (("Without regularization", 0), ("With L2 regularization", 10)):
        weights, history = fit_binary(train, y_train, learning_rate=0.15, steps=2200, penalty=penalty)
        probabilities = sigmoid(np.column_stack([np.ones(len(test)), test]) @ weights)
        models[label] = {"weights": weights, "history": history,
                         "test_accuracy": float(accuracy_score(y_test, probabilities >= 0.5))}
    result = {label: {"test_accuracy": item["test_accuracy"],
                      "final_training_loss": item["history"][-1]}
              for label, item in models.items()}
    if make_plot:
        xx, yy = np.meshgrid(np.linspace(x[:, 0].min() - 0.3, x[:, 0].max() + 0.3, 160),
                             np.linspace(x[:, 1].min() - 0.2, x[:, 1].max() + 0.2, 160))
        mesh = np.column_stack([xx.ravel(), yy.ravel()])
        mesh_scaled = (polynomial_features(mesh) - mean) / scale
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for axis, (label, item) in zip(axes, models.items()):
            probabilities = sigmoid(np.column_stack([np.ones(len(mesh_scaled)), mesh_scaled]) @ item["weights"])
            axis.contourf(xx, yy, probabilities.reshape(xx.shape), levels=[0, 0.5, 1], alpha=0.22,
                          colors=["#2978b5", "#d67542"])
            axis.contour(xx, yy, probabilities.reshape(xx.shape), levels=[0.5], colors="black", linewidths=1.3)
            axis.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="black", s=25)
            axis.set(title=f"{label}\nTest accuracy {item['test_accuracy']:.3f}",
                     xlabel="Petal length (cm)", ylabel="Petal width (cm)")
        save_figure(PLOT)
    return result


if __name__ == "__main__":
    show_result(9, run(), PLOT)
