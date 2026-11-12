from flask import Flask, abort, render_template, request

import exp_12, exp_13, exp_14, exp_15, exp_16, exp_17, exp_18, exp_19, exp_20, exp_21
from core import BOOKS, BOOK_BY_ID, RATINGS, READERS, get_user_ratings

app = Flask(__name__)
reader_rating_edits = {}

ASSIGNMENTS = {
    12: ("Content-based recommendations", "Match books by their genres and authors.", exp_12.recommend),
    13: ("Collaborative filtering", "Use rating patterns from readers with similar interests.", exp_13.recommend),
    14: ("Cold-start recommendations", "Use a genre preference or catalog popularity when rating history is unavailable.", exp_14.recommend),
    15: ("User-based nearest neighbors", "Find nearby readers and combine their ratings.", exp_15.recommend),
    16: ("Item-based nearest neighbors", "Predict ratings from books with similar reader ratings.", exp_16.recommend),
    17: ("Adjusted cosine similarity", "Compare book ratings after centering each reader's rating scale.", exp_17.recommend),
    18: ("Item-based k-nearest neighbors", "Compare recommendations for books already read and books not yet read.", exp_18.recommend),
    19: ("Hybrid recommendations", "Combine book content, reader similarity, and catalog ratings.", exp_19.recommend),
    20: ("Knowledge-based recommendations", "Rank books against explicit genre, author, and length requirements.", exp_20.recommend),
    21: ("Matrix factorization with SGD", "Estimate an unknown book rating from learned reader and book factors using stochastic gradient descent.", exp_21.factorize),
}


@app.get("/")
def home():
    return render_template("home.html", assignments=ASSIGNMENTS)


@app.route("/assignment/<int:number>", methods=["GET", "POST"])
def assignment(number):
    if number not in ASSIGNMENTS:
        abort(404)
    title, description, method = ASSIGNMENTS[number]
    reader = request.values.get("reader", "R01")
    genre = request.values.get("genre", "Mystery")
    author = request.values.get("author", "")
    status = request.values.get("status", "unseen")
    target_book = request.values.get("book_id", "B03")
    epochs = request.values.get("epochs", "20")
    unknown_books = [book for book in BOOKS if (reader, book["id"]) not in RATINGS]
    if number == 21 and unknown_books and target_book not in {book["id"] for book in unknown_books}:
        target_book = unknown_books[0]["id"]
    weights = {name: request.values.get(name, default) for name, default in (
        ("genre_weight", "0.60"), ("author_weight", "0.25"), ("length_weight", "0.15"))}
    hybrid_weights = {name: request.values.get(name, default) for name, default in (
        ("content_weight", "0.50"), ("collaborative_weight", "0.35"), ("popularity_weight", "0.15"))}
    error = None
    factorization = None

    if number in {12, 13, 16}:
        if reader not in READERS:
            abort(404)
        current_ratings = reader_rating_edits.setdefault(reader, get_user_ratings(reader).copy())
        if request.method == "POST":
            action = request.form.get("action")
            try:
                book_id = request.form.get("book_id", "")
                rating = int(request.form.get("rating", ""))
                if book_id not in BOOK_BY_ID or rating not in range(1, 6):
                    raise ValueError("Choose a catalog book and a rating from 1 to 5.")
                if action == "update_rating" and book_id not in current_ratings:
                    raise ValueError("Choose a book already listed for this reader.")
                if action == "add_book" and book_id in current_ratings:
                    raise ValueError("That book is already in this reader's list.")
                if action not in {"update_rating", "add_book"}:
                    raise ValueError("Choose a valid rating action.")
                current_ratings[book_id] = rating
            except ValueError as exc:
                error = str(exc)
        try:
            if number == 12:
                rows = exp_12.recommend(reader=reader, ratings=current_ratings)
            elif number == 13:
                rows = exp_13.recommend(reader=reader, reader_ratings=current_ratings)
            else:
                rows = exp_16.recommend(reader=reader, reader_ratings=current_ratings)
        except (ValueError, KeyError) as exc:
            rows, error = [], str(exc)
        rated_books = [{**BOOK_BY_ID[book], "rating": rating}
                       for book, rating in current_ratings.items()]
        available_books = [book for book in BOOKS if book["id"] not in current_ratings]
        return render_template("assignment.html", number=number, title=title, description=description,
                               rows=rows, reader=reader, readers=READERS, genres=sorted(
                                   {genre for book in BOOKS for genre in book["genres"]}),
                               author=author, status=status, error=error, rated_books=rated_books,
                               available_books=available_books)

    try:
        if number == 14:
            rows = method(reader=None, genres=[genre] if genre else None)
        elif number == 18:
            rows = method(reader=reader, status=status)
        elif number == 20:
            pages = int(request.values.get("max_pages", "500"))
            rows = method(genre=genre, author=author, max_pages=pages,
                          **{name: float(value) for name, value in weights.items()})
        elif number == 19:
            rows = method(reader=reader,
                          **{name: float(value) for name, value in hybrid_weights.items()})
        elif number == 21:
            factorization = method(reader=reader, book_id=target_book, epochs=int(epochs))
            rows = []
        else:
            rows = method(reader=reader)
        error = None
    except (ValueError, KeyError) as exc:
        rows, error = [], str(exc)
    return render_template("assignment.html", number=number, title=title, description=description,
                           rows=rows, reader=reader, readers=READERS, genres=sorted(
                               {genre for book in BOOKS for genre in book["genres"]}),
                           authors=sorted({book["author"] for book in BOOKS}),
                           books=BOOKS, unknown_books=unknown_books,
                           target_book=target_book, epochs=epochs,
                           factorization=factorization,
                           author=author, genre=genre, status=status, error=error, weights=weights,
                           hybrid_weights=hybrid_weights,
                           max_pages=request.values.get("max_pages", "500"))


if __name__ == "__main__":
    app.run(debug=False, port=5002)
