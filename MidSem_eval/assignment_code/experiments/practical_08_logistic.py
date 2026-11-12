from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from core import SEED, fit_binary, fit_multiclass, sigmoid, standardize
from experiments.common import plt, save_figure, show_result


PLOT = "task_8_logistic.png"


def run(make_plot=False):
    # Train binary and three class logistic models and compare their held out predictions.
    iris = load_iris()
    binary_labels = (iris.target == 0).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data[:, [2, 3]], binary_labels, test_size=0.2,
        stratify=binary_labels, random_state=SEED
    )
    train, test, _, _ = standardize(x_train, x_test)
    binary_weights, binary_history = fit_binary(train, y_train)
    binary_probabilities = sigmoid(np.column_stack([np.ones(len(test)), test]) @ binary_weights)
    binary_accuracy = float(accuracy_score(y_test, binary_probabilities >= 0.5))

    x_train, x_test, y_train, y_test = train_test_split(
        iris.data[:, [2, 3]], iris.target, test_size=0.2,
        stratify=iris.target, random_state=SEED
    )
    train, test, _, _ = standardize(x_train, x_test)
    multiclass_weights, multiclass_history = fit_multiclass(train, y_train)
    scores = np.column_stack([np.ones(len(test)), test]) @ multiclass_weights
    multiclass_accuracy = float(accuracy_score(y_test, np.argmax(scores, axis=1)))
    result = {"binary_test_accuracy": binary_accuracy,
              "multiclass_test_accuracy": multiclass_accuracy,
              "binary_initial_loss": binary_history[0], "binary_final_loss": binary_history[-1],
              "multiclass_initial_loss": multiclass_history[0], "multiclass_final_loss": multiclass_history[-1]}
    if make_plot:
        plt.figure(figsize=(7, 3.5))
        plt.plot(binary_history, label=f"Binary ({binary_accuracy:.2f} test accuracy)")
        plt.plot(multiclass_history, label=f"Three class ({multiclass_accuracy:.2f} test accuracy)")
        plt.xlabel("Gradient step")
        plt.ylabel("Training cross entropy")
        plt.title("Manual logistic regression training")
        plt.legend()
        save_figure(PLOT)
    return result


if __name__ == "__main__":
    show_result(8, run(), PLOT)
