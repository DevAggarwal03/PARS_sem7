import json

from flask import Flask, abort, render_template, request

from core import (
    ACTIONS, CLASSES, LOSSES, ROOT, audio_clips, audio_features, bayes_iris, dataset,
    dataset_performance, minimum_risk, store_datasets, text_classify, text_steps,
)


app = Flask(__name__)

IRIS_DESCRIPTIONS = [
    "Sepal length in centimetres", "Sepal width in centimetres",
    "Petal length in centimetres", "Petal width in centimetres",
]
WINE_DESCRIPTIONS = [
    "Alcohol concentration", "Malic acid concentration", "Ash content",
    "Ash alkalinity", "Magnesium concentration", "Total phenols",
    "Flavonoid phenols", "Nonflavonoid phenols", "Proanthocyanidins",
    "Colour intensity", "Hue", "Diluted wine absorbance ratio", "Proline concentration",
]


@app.route("/")
def home():
    # Introduce every assignment demonstration and link each page to its matching practical number.
    return render_template("home.html")


@app.route("/datasets")
def datasets():
    # Show stored source datasets, feature descriptions, and measured classification performance.
    store_datasets()
    entries = []
    for name, descriptions in (("iris", IRIS_DESCRIPTIONS), ("wine", WINE_DESCRIPTIONS)):
        source = dataset(name)
        entries.append({"name": name.title(), "rows": len(source.data),
                        "classes": ", ".join(map(str, source.target_names)),
                        "features": list(zip(source.feature_names, descriptions)),
                        "preview": source.data[:4].round(2).tolist(),
                        "score": dataset_performance(name)["accuracy"]})
    return render_template("datasets.html", entries=entries)


@app.route("/audio")
def audio():
    # Present twelve playable recordings and the selected clip's five timestamped feature rows.
    clips = audio_clips()
    selected_id = request.args.get("clip", clips[0]["id"])
    if selected_id not in {clip["id"] for clip in clips}:
        abort(404)
    summary = []
    selected_rows = []
    for clip in clips:
        rows = audio_features(clip["id"])
        summary.append({**clip, "max_amplitude": sum(row["max_amplitude"] for row in rows) / len(rows),
                        "rms": sum(row["rms"] for row in rows) / len(rows),
                        "zcr": sum(row["zcr"] for row in rows) / len(rows),
                        "centroid": sum(row["centroid"] for row in rows) / len(rows)})
        if clip["id"] == selected_id:
            selected_rows = rows
    selected = next(clip for clip in summary if clip["id"] == selected_id)
    return render_template("audio.html", clips=summary, selected=selected, rows=selected_rows)


@app.route("/bayes", methods=["GET", "POST"])
def bayes():
    # Classify submitted Iris measurements using either one or several input features.
    mode = request.form.get("mode", "single")
    values = request.form.get("values", "1.3" if mode == "single" else "4.4,3.0,1.3,0.2")
    result = None
    error = None
    if request.method == "POST":
        try:
            parsed = [float(item.strip()) for item in values.split(",")]
            indices = [2] if mode == "single" else [0, 1, 2, 3]
            if len(parsed) != len(indices):
                raise ValueError(f"Expected {len(indices)} value(s)")
            result = bayes_iris(parsed, indices)
        except ValueError as exc:
            error = str(exc)
    return render_template("bayes.html", mode=mode, values=values, result=result, error=error)


@app.route("/text", methods=["GET", "POST"])
def text():
    # Show text preparation stages and a separately classified user supplied test sentence.
    paragraph = request.form.get("paragraph", "The team scored a goal. The coach planned the game.")
    test_sentence = request.form.get("test_sentence", "A player won the football match")
    steps = text_steps(paragraph)
    result = text_classify(test_sentence) if request.method == "POST" else None
    return render_template("text.html", paragraph=paragraph, test_sentence=test_sentence,
                           steps=steps, result=result)


@app.route("/risk", methods=["GET", "POST"])
def risk():
    # Display each possible action risk and visibly trigger the minimum risk choice.
    values = [request.form.get(name, default) for name, default in zip(CLASSES, ("0.2", "0.3", "0.5"))]
    error = None
    try:
        result = minimum_risk([float(value) for value in values])
    except ValueError as exc:
        result = None
        error = str(exc)
    return render_template("risk.html", classes=CLASSES, actions=ACTIONS, losses=LOSSES,
                           values=values, result=result, error=error)


@app.route("/practical/<int:number>")
def practical(number):
    # Display saved regression or regularization results on their three existing practical pages.
    if number not in range(7, 10):
        abort(404)
    with (ROOT / "results.json").open() as stream:
        result = json.load(stream)[f"task_{number}"]
    return render_template("experiment.html", number=number, result=result)


@app.route("/practical/10")
def pca_page():
    # Present the PCA projection, held out prediction, and saved plot on its page.
    with (ROOT / "results.json").open() as stream:
        result = json.load(stream)["task_10"]
    return render_template("projection.html", number=10, label="PCA",
                           title="X — Principal component analysis",
                           aim="Implement Principal component analysis (PCA) for any application which involves testing. Also show the projected line and new reduced feature mapped on projected line.",
                           method="PCA chooses the direction with the greatest overall variance without using species labels.",
                           plot="task_10_pca.png", result=result)


@app.route("/practical/11")
def lda_page():
    # Present the LDA projection, held out prediction, and saved plot on its page.
    with (ROOT / "results.json").open() as stream:
        result = json.load(stream)["task_11"]
    return render_template("projection.html", number=11, label="LDA",
                           title="XI — Linear discriminant analysis",
                           aim="Implement Linear Discriminant analysis (LDA) for any application which involves testing. Also show the projected line and new reduced feature mapped on projected line.",
                           method="LDA uses training species labels to favour separation between classes relative to spread within classes.",
                           plot="task_11_lda.png", result=result)


if __name__ == "__main__":
    app.run(debug=False, port=5001)
