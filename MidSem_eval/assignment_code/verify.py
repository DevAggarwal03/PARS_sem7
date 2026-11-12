import json
import hashlib
import wave

import numpy as np

from app import app
from core import (DATA, MEDIA, ROOT, audio_clips, audio_features, bayes_iris,
                  minimum_risk, text_classify, text_steps, window_features)


def main():
    # Check numerical requirements and exercise each interactive page with representative inputs.
    results = json.loads((ROOT / "results.json").read_text())
    clips = audio_clips()
    assert len(clips) == results["task_3"]["recordings"] == 12
    assert len({clip["source_id"] for clip in clips}) == 12
    assert results["task_3"]["total_windows"] == 60
    for clip in clips:
        original = (DATA / "audio_sources" / clip["filename"]).read_bytes()
        playback = (MEDIA / "esc50" / clip["filename"]).read_bytes()
        assert hashlib.sha256(original).hexdigest() == clip["source_sha256"]
        assert hashlib.sha256(playback).hexdigest() == clip["playback_sha256"]
        with wave.open(str(MEDIA / "esc50" / clip["filename"]), "rb") as source:
            assert (source.getframerate(), source.getnchannels(), source.getnframes()) == (16000, 1, 80000)
        rows = audio_features(clip["id"])
        assert len(rows) == 5
        assert [(row["start"], row["end"]) for row in rows] == [(i, i + 1) for i in range(5)]
        assert all(np.isfinite([row["rms"], row["zcr"], row["centroid"]]).all()
                   and row["rms"] >= 0 and 0 <= row["zcr"] <= 1
                   and 0 <= row["centroid"] <= 8000 for row in rows)
    silence = window_features(np.zeros(16000), 16000)
    assert silence == {"rms": 0.0, "zcr": 0.0, "centroid": 0.0}
    tone = window_features(np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000), 16000)
    assert abs(tone["rms"] - np.sqrt(0.5)) < 1e-8
    assert abs(tone["zcr"] - 0.125) < 0.001
    assert abs(tone["centroid"] - 1000) < 1e-6
    assert results["task_7"]["final_loss"] < results["task_7"]["initial_loss"]
    assert results["task_8"]["binary_final_loss"] < results["task_8"]["binary_initial_loss"]
    assert results["task_8"]["multiclass_final_loss"] < results["task_8"]["multiclass_initial_loss"]
    for key in ("task_10", "task_11"):
        item = results[key]
        assert 0 <= item["test_accuracy"] <= 1
        assert len(item["mapped_point"]) == 2
    assert bayes_iris([1.3], [2])["prediction"] == "setosa"
    assert bayes_iris([4.4, 3.0, 1.3, 0.2], [0, 1, 2, 3])["prediction"] == "setosa"
    assert len(text_steps("The team scored. The coach planned.")["sentences"]) == 2
    assert text_classify("A player won the football match")["prediction"] == "Sports"
    assert minimum_risk([0.2, 0.3, 0.5])["action"] == "Stop machine"
    with app.test_client() as client:
        for path in ("/", "/datasets", "/audio", "/bayes", "/text", "/risk"):
            assert client.get(path).status_code == 200
        for number, plot in ((7, "linear"), (8, "logistic"), (9, "regularization"),
                             (10, "pca"), (11, "lda")):
            page = client.get(f"/practical/{number}")
            assert page.status_code == 200
            assert f"task_{number}_{plot}.png".encode() in page.data
            assert (ROOT / "static" / "plots" / f"task_{number}_{plot}.png").is_file()
        assert client.get("/practical/12").status_code == 404
        assert b"Predicted: setosa" in client.post("/bayes", data={"mode": "single", "values": "1.3"}).data
        assert b"Predicted: setosa" in client.post("/bayes", data={"mode": "multiple", "values": "4.4,3.0,1.3,0.2"}).data
        assert b"Test sentence: Sports" in client.post("/text", data={"paragraph": "The team scored. The coach planned.", "test_sentence": "A player won the football match"}).data
        assert b"Triggered action: Stop machine" in client.post("/risk", data={"Normal": "0.2", "Warning": "0.3", "Fault": "0.5"}).data
        for clip in clips:
            page = client.get(f"/audio?clip={clip['id']}")
            assert page.status_code == 200 and clip["filename"].encode() in page.data
            assert client.get(f"/static/media/esc50/{clip['filename']}").status_code == 200
        assert client.get("/audio?clip=unknown").status_code == 404
    print("All numerical and interactive checks passed.")


if __name__ == "__main__":
    main()
