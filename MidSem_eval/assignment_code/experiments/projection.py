import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from core import SEED, nearest_centroid
from experiments.common import plt, save_figure


def projection_data():
    # Give PCA and LDA the same labelled flowers and reproducible held out samples.
    ds = load_iris()
    x = ds.data[:, [2, 3]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, ds.target, test_size=0.2, stratify=ds.target, random_state=SEED
    )
    return ds, x_train, x_test, y_train, y_test


def finish_projection(data, center, axis, label, plot_name, make_plot=False):
    # Project held out flowers, classify nearest class centres, and optionally refresh the figure.
    ds, x_train, x_test, y_train, y_test = data
    train_projection = (x_train - center) @ axis
    test_projection = (x_test - center) @ axis
    predictions = nearest_centroid(train_projection, y_train, test_projection)
    example = x_test[0]
    coordinate = float(test_projection[0])
    mapped = center + coordinate * axis
    result = {"test_accuracy": float(accuracy_score(y_test, predictions)),
              "test_sample": example.tolist(), "reduced_coordinate": coordinate,
              "mapped_point": mapped.tolist(), "actual_class": str(ds.target_names[y_test[0]]),
              "predicted_class": str(ds.target_names[predictions[0]])}
    if make_plot:
        plt.figure(figsize=(6, 4.5))
        for class_id in range(3):
            group = x_train[y_train == class_id]
            plt.scatter(group[:, 0], group[:, 1], s=20, alpha=0.65,
                        label=ds.target_names[class_id])
        line = np.linspace(-3, 3, 2)
        points = center + line[:, None] * axis
        plt.plot(points[:, 0], points[:, 1], color="black", label=f"{label} projected line")
        plt.scatter([example[0]], [example[1]], marker="*", s=190, color="red", label="Held-out sample")
        plt.scatter([mapped[0]], [mapped[1]], marker="x", s=130, color="black", label="Mapped point")
        plt.plot([example[0], mapped[0]], [example[1], mapped[1]], color="red", linestyle="--")
        plt.xlabel("Petal length (cm)")
        plt.ylabel("Petal width (cm)")
        plt.title(f"{label}: test sample mapped to one dimension")
        plt.legend(fontsize=7)
        save_figure(plot_name)
    return result
