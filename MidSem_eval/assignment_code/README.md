# PARS MidSem assignment code

This project contains the runnable programs for practicals II-XI. Practical I is the dataset study in the report. The `data` folder contains stored copies of the Iris and Wine datasets. Practical III uses twelve real five-second machine recordings from ESC-50, with their original sources and attribution in `data/audio_manifest.json`.

## Setup

From this folder:

```bash
uv sync
uv run python prepare_audio.py
uv run python run_experiments.py
uv run python app.py
```

Open `http://127.0.0.1:5001/`. The Flask pages cover practicals II-XI. The audio page lists all twelve recordings and shows five one-second measurements for the selected clip. The Practical VII-XI pages display saved plots and results from the experiment runner, which refreshes the Practical III plots and produces `results.json` in the project folder.

## Run one experiment without Flask

From the `assignment_code` folder, run the file requested by the instructor:

```bash
uv run python experiments/practical_07_linear.py
uv run python experiments/practical_08_logistic.py
uv run python experiments/practical_09_regularization.py
uv run python experiments/practical_10_pca.py
uv run python experiments/practical_11_lda.py
```

Each command trains or evaluates its model, prints fresh results, and opens its existing saved plot. It does not change the plot or `results.json`. Run `uv run python run_experiments.py` only when you want to refresh the saved figures and results for the Flask pages.

## Methods

- Iris and Wine are distributed in scikit-learn from the [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris) and [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine) benchmarks.
- Practical II uses Gaussian Naive Bayes and stratified 80/20 train/test splits with seed 42. The literature results in the report use different protocols and are not directly comparable.
- Practical III uses twelve independent ESC-50 source recordings from washing machines, vacuum cleaners, chainsaws, engines, helicopters, and trains. `prepare_audio.py` downloads the official 44.1 kHz files and resamples playback copies to 16 kHz; it preserves source URLs, original Freesound attribution, clip licenses, and SHA-256 checksums. The collection is [CC BY-NC 3.0](https://github.com/karolpiczak/ESC-50#license). Each five-second clip yields five one-second windows, giving sixty timestamped feature rows. RMS, zero-crossing rate, and spectral centroid describe recorded sound patterns; they do not diagnose faults. The original source files remain in `data/audio_sources` and the playable copies remain in `static/media/esc50`.
- Practicals VII-VIII implement the regression training loops directly with NumPy arithmetic. No library regression trainer is called. Practical IX reuses the manually coded binary logistic trainer with an L2 penalty.
- Practicals X-XI fit their projection directions on training data only, then reduce and classify held-out data with nearest centroids.

Run checks with `uv run python verify.py`. The audio preparation step needs internet access only when original source files are missing.
