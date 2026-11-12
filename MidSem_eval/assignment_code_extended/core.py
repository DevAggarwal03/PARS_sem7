import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
with (ROOT / "catalog.json").open(encoding="utf-8") as source:
    BOOKS = json.load(source)
with (ROOT / "ratings.csv").open(encoding="utf-8") as source:
    RATINGS = {(row["reader_id"], row["book_id"]): float(row["rating"])
               for row in csv.DictReader(source)}
READERS = sorted({reader for reader, _ in RATINGS})
BOOK_BY_ID = {book["id"]: book for book in BOOKS}
BOOK_IDS = [book["id"] for book in BOOKS]
MATRIX = np.array([[RATINGS.get((reader, book), np.nan) for book in BOOK_IDS]
                   for reader in READERS], dtype=float)


def get_user_ratings(reader):
    return {book: rating for (owner, book), rating in RATINGS.items() if owner == reader}


def popular_scores():
    scores = {}
    for book in BOOK_IDS:
        observed = [rating for (reader, item), rating in RATINGS.items() if item == book]
        scores[book] = float(np.mean(observed)) if observed else 0.0
    return scores


def book_vector(book_id):
    book = BOOK_BY_ID[book_id]
    tokens = {f"genre:{genre}" for genre in book["genres"]}
    tokens.add(f"author:{book['author']}")
    return tokens


def cosine(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denom) if denom else 0.0


def book_similarity(first, second, adjusted=False, reader=None, reader_ratings=None):
    i, j = BOOK_IDS.index(first), BOOK_IDS.index(second)
    left, right = MATRIX[:, i].copy(), MATRIX[:, j].copy()
    if reader is not None and reader_ratings is not None and reader in READERS:
        row = READERS.index(reader)
        left[row] = reader_ratings.get(first, np.nan)
        right[row] = reader_ratings.get(second, np.nan)
    shared = ~np.isnan(left) & ~np.isnan(right)
    if shared.sum() < 1:
        return 0.0
    a, b = left[shared].copy(), right[shared].copy()
    if adjusted:
        row_means = np.nanmean(MATRIX, axis=1)
        users = np.where(shared)[0]
        a -= row_means[users]
        b -= row_means[users]
    return cosine(a, b)


def user_similarity(first, second, first_ratings=None):
    first_ratings = first_ratings if first_ratings is not None else get_user_ratings(first)
    a = np.array([first_ratings.get(book, np.nan) for book in BOOK_IDS])
    b = np.array([RATINGS.get((second, book), np.nan) for book in BOOK_IDS])
    shared = ~np.isnan(a) & ~np.isnan(b)
    if shared.sum() < 2:
        return 0.0
    a, b = a[shared], b[shared]
    a, b = a - a.mean(), b - b.mean()
    return cosine(a, b)


def predict_item(reader, candidate, adjusted=False, neighbors=5, reader_ratings=None):
    known = reader_ratings if reader_ratings is not None else get_user_ratings(reader)
    weighted, total = 0.0, 0.0
    for seen, rating in known.items():
        similarity = book_similarity(candidate, seen, adjusted=adjusted,
                                     reader=reader, reader_ratings=reader_ratings)
        if similarity > 0:
            weighted += similarity * rating
            total += similarity
    baseline = np.mean(list(known.values())) if known else np.mean(list(RATINGS.values()))
    if total == 0:
        return float(baseline)
    return float(weighted / total)


def result_rows(scores, exclude=(), limit=5):
    excluded = set(exclude)
    ordered = sorted(((book, float(score)) for book, score in scores.items()
                      if book in BOOK_BY_ID and book not in excluded),
                     key=lambda item: (-item[1], BOOK_BY_ID[item[0]]["title"]))[:limit]
    return [{**BOOK_BY_ID[book], "score": round(score, 4)} for book, score in ordered]
