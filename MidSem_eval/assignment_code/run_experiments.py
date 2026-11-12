import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import (
    PLOTS, ROOT, audio_clips, audio_features, dataset_performance, ensure_dirs,
    store_datasets,
)
from experiments.practical_07_linear import run as run_linear
from experiments.practical_08_logistic import run as run_logistic
from experiments.practical_09_regularization import run as run_regularization
from experiments.practical_10_pca import run as run_pca
from experiments.practical_11_lda import run as run_lda


def save_figure(name):
    # Save a clear reproducible figure for the report and close its plotting canvas.
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=180, bbox_inches="tight")
    plt.close()


def dataset_plot(results):
    # Compare held out Gaussian Bayes accuracy for the two selected UCI datasets.
    labels = ["Iris", "Wine"]
    scores = [results["task_2"][name]["accuracy"] for name in ("iris", "wine")]
    plt.figure(figsize=(6, 3.5))
    bars = plt.bar(labels, scores, color=["#3479ae", "#b46d30"])
    plt.ylim(0, 1.08)
    plt.ylabel("Held-out accuracy")
    plt.title("Gaussian Naive Bayes, stratified 80/20 split")
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.3f}", ha="center")
    save_figure("task_2_accuracy.png")


def audio_plot(rows, clip):
    # Plot the four measured features across five timestamped windows of one real recording.
    fig, axes = plt.subplots(4, 1, figsize=(7, 6.8), sharex=True)
    for axis, key, label in zip(axes, ("max_amplitude", "rms", "zcr", "centroid"),
                                ("Maximum amplitude", "RMS energy", "Zero crossing rate",
                                 "Spectral centroid (Hz)")):
        axis.plot([row["start"] for row in rows], [row[key] for row in rows], marker="o")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel("Window start (seconds)")
    fig.suptitle(f"{clip['label']}: {clip['filename']} (ESC-50)")
    save_figure(f"task_3_{clip['id']}.png")


def main():
    # Produce all tested numerical results and charts from one reproducible entry point.
    ensure_dirs()
    store_datasets()
    results = {"task_2": {name: dataset_performance(name) for name in ("iris", "wine")}}
    dataset_plot(results)
    clips = []
    for clip in audio_clips():
        rows = audio_features(clip["id"])
        clips.append({"id": clip["id"], "category": clip["category"],
                      "source_id": clip["source_id"], "windows": rows,
                      "mean_max_amplitude": float(np.mean([row["max_amplitude"] for row in rows])),
                      "mean_rms": float(np.mean([row["rms"] for row in rows])),
                      "mean_zcr": float(np.mean([row["zcr"] for row in rows])),
                      "mean_centroid": float(np.mean([row["centroid"] for row in rows]))})
        audio_plot(rows, clip)
    results["task_3"] = {"recordings": len(clips), "windows_per_recording": 5,
                         "total_windows": sum(len(clip["windows"]) for clip in clips),
                         "clips": clips}
    results["task_7"] = run_linear(make_plot=True)
    results["task_8"] = run_logistic(make_plot=True)
    results["task_9"] = run_regularization(make_plot=True)
    results["task_10"] = run_pca(make_plot=True)
    results["task_11"] = run_lda(make_plot=True)
    (ROOT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
