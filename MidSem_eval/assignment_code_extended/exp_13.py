"""Collaborative filtering with centered reader similarities."""
import numpy as np

from core import BOOK_IDS, READERS, get_user_ratings, result_rows, user_similarity, RATINGS


def recommend(reader="R01", limit=5, reader_ratings=None):
    known = reader_ratings if reader_ratings is not None else get_user_ratings(reader)
    scores = {}
    for book in BOOK_IDS:
        if book in known:
            continue
        neighbors = [(user_similarity(reader, other, first_ratings=known), RATINGS[(other, book)])
                     for other in READERS if other != reader and (other, book) in RATINGS]
        useful = [(sim, rating) for sim, rating in neighbors if sim > 0]
        scores[book] = (sum(sim * rating for sim, rating in useful) / sum(sim for sim, _ in useful)
                        if useful else float(np.mean([rating for _, rating in neighbors]))
                        if neighbors else 0.0)
    return result_rows(scores, limit=limit)


def main():
    print(recommend())


if __name__ == "__main__":
    main()
