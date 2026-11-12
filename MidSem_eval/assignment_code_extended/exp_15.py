"""User-based nearest-neighbor book recommendations."""
import numpy as np

from core import BOOK_IDS, READERS, RATINGS, get_user_ratings, result_rows, user_similarity


def recommend(reader="R01", neighbors=3, limit=5):
    known = get_user_ratings(reader)
    reader_mean = float(np.mean(list(known.values()))) if known else 0.0
    similar = sorted(((user_similarity(reader, other), other) for other in READERS if other != reader),
                     reverse=True)[:neighbors]
    scores = {}
    for book in BOOK_IDS:
        if book in known:
            continue
        votes = [(sim, RATINGS[(other, book)], float(np.mean(list(get_user_ratings(other).values()))))
                 for sim, other in similar
                 if sim > 0 and (other, book) in RATINGS]
        if votes:
            deviation = sum(sim * (rating - neighbor_mean)
                            for sim, rating, neighbor_mean in votes) / sum(sim for sim, _, _ in votes)
            scores[book] = reader_mean + deviation
        else:
            scores[book] = 0.0
    return result_rows(scores, limit=limit)


def main():
    print(recommend())


if __name__ == "__main__":
    main()
