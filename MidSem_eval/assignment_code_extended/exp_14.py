"""Cold-start recommendations using reader preferences or catalog popularity."""
from core import BOOKS, BOOK_IDS, get_user_ratings, popular_scores, result_rows


def recommend(reader=None, genres=None, limit=5):
    known = get_user_ratings(reader) if reader else {}
    if genres:
        selected = set(genres)
        pool = [book for book in BOOK_IDS if selected.intersection(
            next(item["genres"] for item in BOOKS if item["id"] == book))]
        if pool:
            scores = {book: popular_scores()[book] for book in pool}
            return result_rows(scores, exclude=known, limit=limit)
    return result_rows(popular_scores(), exclude=known, limit=limit)


def main():
    print(recommend())


if __name__ == "__main__":
    main()
