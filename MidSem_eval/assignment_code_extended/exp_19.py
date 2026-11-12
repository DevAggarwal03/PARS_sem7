"""Hybrid recommendations combining book content and reader ratings."""
from core import BOOK_IDS, book_vector, get_user_ratings, popular_scores, result_rows, user_similarity, READERS, RATINGS


def recommend(reader="R01", limit=5, content_weight=0.50,
              collaborative_weight=0.35, popularity_weight=0.15):
    weight_total = content_weight + collaborative_weight + popularity_weight
    if min(content_weight, collaborative_weight, popularity_weight) < 0 or weight_total <= 0:
        raise ValueError("Weights must be nonnegative and at least one must be greater than zero.")
    content_weight, collaborative_weight, popularity_weight = (
        weight / weight_total for weight in (content_weight, collaborative_weight, popularity_weight)
    )
    known = get_user_ratings(reader)
    liked = set().union(*(book_vector(book) for book, rating in known.items() if rating >= 4)) if known else set()
    popularity = popular_scores()
    scores = {}
    components = {}
    for book in BOOK_IDS:
        if book in known:
            continue
        features = book_vector(book)
        content = len(liked & features) / max(1, len(liked | features))
        votes = [(user_similarity(reader, other), RATINGS[(other, book)])
                 for other in READERS if other != reader and (other, book) in RATINGS]
        collaborative = (sum(sim * rating for sim, rating in votes if sim > 0)
                         / sum(sim for sim, rating in votes if sim > 0)) if any(sim > 0 for sim, _ in votes) else 0.0
        popularity_score = popularity[book]
        scores[book] = (content_weight * content * 5
                        + collaborative_weight * collaborative
                        + popularity_weight * popularity_score)
        components[book] = {"content_score": content,
                            "collaborative_score": collaborative,
                            "popularity_score": popularity_score}
    rows = result_rows(scores, limit=limit)
    for row in rows:
        row.update({name: round(value, 4) for name, value in components[row["id"]].items()})
    return rows


def main():
    print(recommend())


if __name__ == "__main__":
    main()
