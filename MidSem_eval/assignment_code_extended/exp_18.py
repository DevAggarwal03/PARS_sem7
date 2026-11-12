"""Item-based k-nearest-neighbor predictions for books already read or not read."""
from core import BOOK_IDS, BOOK_BY_ID, BOOKS, BOOK_IDS, RATINGS, get_user_ratings, book_similarity, result_rows


def recommend(reader="R01", status="unseen", neighbors=3, limit=5):
    known = get_user_ratings(reader)
    scores = {}
    for candidate in BOOK_IDS:
        is_seen = candidate in known
        if (status == "seen") != is_seen:
            continue
        rated = known if is_seen else known
        closest = sorted(((book_similarity(candidate, seen), rating)
                          for seen, rating in rated.items() if seen != candidate), reverse=True)[:neighbors]
        weight_sum = sum(max(0.0, similarity) for similarity, _ in closest)
        scores[candidate] = (sum(max(0.0, similarity) * rating for similarity, rating in closest) / weight_sum
                             if weight_sum else known.get(candidate, 0.0))
    return result_rows(scores, limit=limit)


def main():
    print(recommend())


if __name__ == "__main__":
    main()
