"""Item-based nearest neighbors with rating prediction."""
from core import BOOK_IDS, get_user_ratings, predict_item, result_rows


def recommend(reader="R01", limit=5, reader_ratings=None):
    known = reader_ratings if reader_ratings is not None else get_user_ratings(reader)
    scores = {book: predict_item(reader, book, reader_ratings=known)
              for book in BOOK_IDS if book not in known}
    return result_rows(scores, limit=limit)


def main():
    print(recommend())


if __name__ == "__main__":
    main()
