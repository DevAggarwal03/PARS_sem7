import csv
import hashlib
import io
import json
import re
import urllib.request
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from core import DATA, MEDIA


BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
SOURCES = {
    "washing_machine": ["1-21896-A-35.wav", "1-23996-A-35.wav"],
    "vacuum_cleaner": ["1-100210-A-36.wav", "1-19840-A-36.wav"],
    "chainsaw": ["1-116765-A-41.wav", "1-19898-A-41.wav"],
    "engine": ["1-18527-A-44.wav", "1-22882-A-44.wav"],
    "helicopter": ["1-172649-A-40.wav", "1-181071-A-40.wav"],
    "train": ["1-119125-A-45.wav", "2-122067-A-45.wav"],
}


def fetch(url):
    # Download each original recording and its attribution from the official ESC-50 repository.
    with urllib.request.urlopen(url, timeout=40) as response:
        return response.read()


def digest(data):
    # Record a SHA-256 checksum so every source and playback file can be verified.
    return hashlib.sha256(data).hexdigest()


def pcm16(data):
    # Decode mono signed PCM data and verify each recording has the documented source format.
    with wave.open(io.BytesIO(data), "rb") as stream:
        rate = stream.getframerate()
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        frames = stream.getnframes()
        samples = np.frombuffer(stream.readframes(frames), dtype="<i2").astype(np.float64) / 32768
    if (rate, channels, width, frames) != (44100, 1, 2, 220500):
        raise ValueError(f"Unexpected WAV format: {rate}, {channels}, {width}, {frames}")
    if np.sqrt(np.mean(samples * samples)) < 0.001:
        raise ValueError("Recording is nearly silent")
    return samples


def attribution(license_text, filename):
    # Preserve the original Freesound title, creator, link, and individual clip licence.
    original_name = filename.rsplit("-", 1)[0] + ".ogg"
    line = next((line for line in license_text.splitlines()
                 if line.startswith(f"- [{original_name}]:")), None)
    if line is None:
        raise ValueError(f"Missing attribution for {filename}")
    match = re.search(r"clip derived from (.*?) \((https?://[^)]+)\) by (.*?) \[([^]]+)\]", line)
    if match is None:
        raise ValueError(f"Unrecognized attribution for {filename}: {line}")
    title, url, author, license_name = match.groups()
    return {"original_title": title, "original_url": url, "author": author,
            "clip_license": license_name}


def main():
    # Build twelve reproducible five-second clips and preserve their complete source information.
    source_dir = DATA / "audio_sources"
    playback_dir = MEDIA / "esc50"
    source_dir.mkdir(parents=True, exist_ok=True)
    playback_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = source_dir / "esc50.csv"
    license_path = source_dir / "LICENSE"
    if not metadata_path.exists():
        metadata_path.write_bytes(fetch(f"{BASE}/meta/esc50.csv"))
    if not license_path.exists():
        license_path.write_bytes(fetch(f"{BASE}/LICENSE"))
    metadata = list(csv.DictReader(metadata_path.read_text().splitlines()))
    by_name = {row["filename"]: row for row in metadata}
    license_text = license_path.read_text()
    manifest = []
    original_ids = set()
    for category, filenames in SOURCES.items():
        for filename in filenames:
            row = by_name[filename]
            if row["category"] != category or row["src_file"] in original_ids:
                raise ValueError(f"Incorrect category or duplicate original: {filename}")
            original_ids.add(row["src_file"])
            source_url = f"{BASE}/audio/{filename}"
            original_path = source_dir / filename
            if not original_path.exists():
                original_path.write_bytes(fetch(source_url))
            original = original_path.read_bytes()
            samples = pcm16(original)
            converted = resample_poly(samples, 160, 441)
            if len(converted) != 80000:
                raise ValueError(f"Unexpected converted length: {filename}")
            converted = np.int16(np.clip(converted, -1, 1 - 1 / 32768) * 32768)
            playback_path = playback_dir / filename
            with wave.open(str(playback_path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(converted.tobytes())
            manifest.append({
                "id": filename.removesuffix(".wav"), "filename": filename,
                "category": category, "label": category.replace("_", " ").title(),
                "source_id": row["src_file"], "source_url": source_url,
                "dataset_url": "https://github.com/karolpiczak/ESC-50",
                "dataset_license": "CC BY-NC 3.0", "source_sha256": digest(original),
                "playback_sha256": digest(playback_path.read_bytes()),
                "duration_seconds": 5, "source_rate": 44100, "playback_rate": 16000,
                **attribution(license_text, filename),
            })
            print(f"Prepared {filename}: {category}, Freesound {row['src_file']}")
    (DATA / "audio_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(manifest)} independent recordings.")


if __name__ == "__main__":
    main()
