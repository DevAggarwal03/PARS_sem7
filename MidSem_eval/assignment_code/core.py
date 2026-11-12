from pathlib import Path
import csv
import json
import re
import wave

import numpy as np
import simplemma
from nltk.stem import SnowballStemmer
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MEDIA = ROOT / "static" / "media"
PLOTS = ROOT / "static" / "plots"
SEED = 42
STOP_WORDS = set("a an and are as at be by for from has have in is it of on or that the this to was were with you your".split())
STEMMER = SnowballStemmer("english")
TEXT_TRAINING = [
    ("The team scored a goal and won the football match", "Sports"),
    ("A player served an ace in the tennis final", "Sports"),
    ("The runner trained for a marathon race", "Sports"),
    ("The coach planned a defense for the game", "Sports"),
    ("A new processor makes the computer faster", "Technology"),
    ("The software update fixed the mobile application", "Technology"),
    ("Engineers built a robot with sensors", "Technology"),
    ("The network server stores digital data", "Technology"),
    ("Doctors recommend exercise for a healthy heart", "Health"),
    ("The hospital studied nutrition and disease", "Health"),
    ("A patient received medicine after a diagnosis", "Health"),
    ("Sleep and diet improve overall health", "Health"),
]
LOSSES = np.array([[0, 5, 20], [2, 1, 8], [8, 4, 0]], dtype=float)
ACTIONS = ["Continue operation", "Inspect machine", "Stop machine"]
CLASSES = ["Normal", "Warning", "Fault"]


def ensure_dirs():
    # Keep every reusable input and generated plot inside the single assignment code project.
    for folder in (DATA, MEDIA, PLOTS):
        folder.mkdir(parents=True, exist_ok=True)


def dataset(name):
    # Load a small official benchmark dataset with its original feature and target names.
    if name == "iris":
        return load_iris()
    if name == "wine":
        return load_wine()
    raise ValueError("Choose iris or wine")


def store_datasets():
    # Save both source datasets as CSV files so the web page can reload them.
    ensure_dirs()
    for name in ("iris", "wine"):
        target = DATA / f"{name}.csv"
        ds = dataset(name)
        with target.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([*ds.feature_names, "class"])
            writer.writerows([*row, int(label)] for row, label in zip(ds.data, ds.target))


def dataset_performance(name):
    # Measure Gaussian Bayes accuracy on a reproducible stratified held out dataset split.
    ds = dataset(name)
    x_train, x_test, y_train, y_test = train_test_split(
        ds.data, ds.target, test_size=0.2, stratify=ds.target, random_state=SEED
    )
    model = GaussianNB().fit(x_train, y_train)
    return {
        "dataset": name,
        "train": len(x_train),
        "test": len(x_test),
        "accuracy": float(accuracy_score(y_test, model.predict(x_test))),
        "features": len(ds.feature_names),
        "classes": len(ds.target_names),
    }


def audio_clips():
    # Read the verified source manifest for the twelve independent machine recordings.
    return json.loads((DATA / "audio_manifest.json").read_text())


def window_features(chunk, rate):
    # Calculate peak and average strength, sign changes, and frequency balance for one audio window.
    max_amplitude = float(np.max(np.abs(chunk)))
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    zcr = float(np.mean(np.signbit(chunk[1:]) != np.signbit(chunk[:-1])))
    spectrum = np.abs(np.fft.rfft(chunk))
    frequencies = np.fft.rfftfreq(len(chunk), 1 / rate)
    magnitude = np.sum(spectrum)
    centroid = float(np.sum(frequencies * spectrum) / magnitude) if magnitude else 0.0
    return {"max_amplitude": max_amplitude, "rms": rms, "zcr": zcr, "centroid": centroid}


def audio_features(clip_id):
    # Measure three interpretable audio features across five one-second windows of one approved clip.
    clip = next((item for item in audio_clips() if item["id"] == clip_id), None)
    if clip is None:
        raise ValueError(f"Unknown audio clip: {clip_id}")
    with wave.open(str(MEDIA / "esc50" / clip["filename"]), "rb") as stream:
        rate = stream.getframerate()
        frames = stream.getnframes()
        signal = np.frombuffer(stream.readframes(frames), dtype="<i2").astype(float) / 32768
        if (rate, stream.getnchannels(), stream.getsampwidth(), frames) != (16000, 1, 2, 80000):
            raise ValueError(f"Unexpected playback format: {clip_id}")
    rows = []
    for index in range(len(signal) // rate):
        chunk = signal[index * rate:(index + 1) * rate]
        rows.append({"start": index, "end": index + 1, **window_features(chunk, rate)})
    return rows


def bayes_iris(values, feature_indices):
    # Fit Gaussian Bayes using selected Iris features and classify one submitted sample.
    ds = load_iris()
    x = ds.data[:, feature_indices]
    sample = np.asarray(values, dtype=float).reshape(1, -1)
    x_train, _, y_train, _ = train_test_split(
        x, ds.target, test_size=0.2, stratify=ds.target, random_state=SEED
    )
    model = GaussianNB().fit(x_train, y_train)
    probabilities = model.predict_proba(sample)[0]
    return {"prediction": ds.target_names[int(np.argmax(probabilities))],
            "probabilities": dict(zip(ds.target_names, map(float, probabilities)))}


def text_steps(paragraph):
    # Show each text cleaning stage before scoring a separate test sentence.
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", paragraph.strip()) if item.strip()]
    tokens = re.findall(r"[a-z]+", paragraph.lower())
    filtered = [token for token in tokens if token not in STOP_WORDS]
    lemmas = [simplemma.lemmatize(token, lang="en") for token in filtered]
    stems = [STEMMER.stem(token) for token in filtered]
    return {"sentences": sentences, "tokens": tokens, "filtered": filtered, "lemmas": lemmas, "stems": stems}


def text_classify(paragraph):
    # Train a transparent multinomial Bayes model on labelled demonstration sentences.
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    corpus = [" ".join(text_steps(text)["lemmas"]) for text, _ in TEXT_TRAINING]
    labels = [label for _, label in TEXT_TRAINING]
    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(corpus)
    model = MultinomialNB(alpha=1).fit(x_train, labels)
    x_test = vectorizer.transform([" ".join(text_steps(paragraph)["lemmas"])])
    probabilities = model.predict_proba(x_test)[0]
    return {"prediction": str(model.classes_[int(np.argmax(probabilities))]),
            "probabilities": dict(zip(model.classes_, map(float, probabilities)))}


def minimum_risk(probabilities):
    # Multiply each action loss by class probabilities and select the least risky action.
    posterior = np.asarray(probabilities, dtype=float)
    if np.any(posterior < 0) or not np.isclose(posterior.sum(), 1):
        raise ValueError("Probabilities must be nonnegative and total 1")
    risks = LOSSES @ posterior
    winner = int(np.argmin(risks))
    return {"action": ACTIONS[winner], "risks": dict(zip(ACTIONS, map(float, risks))),
            "posterior": dict(zip(CLASSES, map(float, posterior)))}


def standardize(train, test):
    # Calculate training only means and scales to prevent information leakage into testing.
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1
    return (train - mean) / scale, (test - mean) / scale, mean, scale


def sigmoid(values):
    # Clip extreme scores before the sigmoid calculation to keep probabilities numerically stable.
    return 1 / (1 + np.exp(-np.clip(values, -30, 30)))


def fit_linear(x, y, learning_rate=0.05, steps=1200):
    # Optimize an intercept and slope using the mean squared error gradient.
    design = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros(design.shape[1])
    history = []
    for _ in range(steps):
        residual = design @ weights - y
        history.append(float(np.mean(residual ** 2)))
        weights -= learning_rate * 2 * design.T @ residual / len(y)
    return weights, history


def fit_binary(x, y, learning_rate=0.1, steps=1600, penalty=0):
    # Optimize binary cross entropy directly with gradient descent and optional L2 regularization.
    design = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros(design.shape[1])
    history = []
    for _ in range(steps):
        probabilities = sigmoid(design @ weights)
        loss = -np.mean(y * np.log(probabilities + 1e-12) + (1 - y) * np.log(1 - probabilities + 1e-12))
        loss += penalty * np.sum(weights[1:] ** 2) / (2 * len(y))
        history.append(float(loss))
        gradient = design.T @ (probabilities - y) / len(y)
        gradient[1:] += penalty * weights[1:] / len(y)
        weights -= learning_rate * gradient
    return weights, history


def fit_multiclass(x, y, classes=3, learning_rate=0.1, steps=1800):
    # Optimize softmax cross entropy for all three Iris classes without a fitted library model.
    design = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros((design.shape[1], classes))
    one_hot = np.eye(classes)[y]
    history = []
    for _ in range(steps):
        scores = design @ weights
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        history.append(float(-np.mean(np.log(probabilities[np.arange(len(y)), y] + 1e-12))))
        weights -= learning_rate * design.T @ (probabilities - one_hot) / len(y)
    return weights, history


def polynomial_features(x):
    # Expand two measured features into quadratic terms for the regularization comparison.
    a, b = x[:, 0], x[:, 1]
    return np.column_stack([a, b, a * a, a * b, b * b])


def first_pca_axis(train):
    # Find the training data direction with the greatest measured variance.
    center = train.mean(axis=0)
    covariance = np.cov((train - center).T)
    values, vectors = np.linalg.eigh(covariance)
    return center, vectors[:, int(np.argmax(values))]


def first_lda_axis(train, labels):
    # Find the direction maximizing between class separation relative to within class variation.
    overall = train.mean(axis=0)
    within = np.zeros((train.shape[1], train.shape[1]))
    between = np.zeros_like(within)
    for label in np.unique(labels):
        group = train[labels == label]
        center = group.mean(axis=0)
        within += (group - center).T @ (group - center)
        difference = (center - overall).reshape(-1, 1)
        between += len(group) * difference @ difference.T
    values, vectors = np.linalg.eig(np.linalg.pinv(within) @ between)
    axis = np.real(vectors[:, int(np.argmax(np.real(values)))])
    return overall, axis / np.linalg.norm(axis)


def nearest_centroid(train_values, train_labels, test_values):
    # Predict each reduced test sample from the nearest training class center.
    centers = {int(label): np.mean(train_values[train_labels == label]) for label in np.unique(train_labels)}
    predictions = [min(centers, key=lambda label: abs(value - centers[label])) for value in test_values]
    return np.asarray(predictions, dtype=int)
