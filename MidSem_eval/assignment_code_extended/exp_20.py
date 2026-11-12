"""Knowledge-based recommendations from weighted reader requirements."""
from core import BOOKS, result_rows


def recommend(genre="Mystery", author="", min_pages=0, max_pages=500, limit=5,
              genre_weight=0.60, author_weight=0.25, length_weight=0.15):
    weight_total = genre_weight + author_weight + length_weight
    if min(genre_weight, author_weight, length_weight) < 0 or weight_total <= 0:
        raise ValueError("Weights must be nonnegative and at least one must be greater than zero.")
    genre_weight, author_weight, length_weight = (
        weight / weight_total for weight in (genre_weight, author_weight, length_weight)
    )
    scores = {}
    components = {}
    for book in BOOKS:
        genre_score = 1.0 if genre in book["genres"] else 0.0
        author_score = 1.0 if author and author.casefold() == book["author"].casefold() else 0.0
        length_score = 1.0 if min_pages <= book["pages"] <= max_pages else 0.0
        scores[book["id"]] = (genre_weight * genre_score + author_weight * author_score
                               + length_weight * length_score)
        components[book["id"]] = {"genre_match": genre_score, "author_match": author_score,
                                  "length_match": length_score}
    rows = result_rows(scores, limit=limit)
    for row in rows:
        row.update(components[row["id"]])
    return rows


def main():
    print(recommend())


if __name__ == "__main__":
    main()
