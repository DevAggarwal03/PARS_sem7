"""Content-based book recommendations from genres and authors."""
from core import BOOK_BY_ID, BOOK_IDS, BOOKS, book_vector, get_user_ratings, result_rows


def recommend(reader="R01", limit=5, ratings=None):
    known = ratings if ratings is not None else get_user_ratings(reader)
    profile = set()
    for book, rating in known.items():
        if rating >= 4:
            profile |= book_vector(book)
    scores = {book: len(profile & book_vector(book)) / max(1, len(profile | book_vector(book)))
              for book in BOOK_IDS}
    return result_rows(scores, exclude=known, limit=limit)


def main():
    print(recommend())


if __name__ == "__main__":
    main()
