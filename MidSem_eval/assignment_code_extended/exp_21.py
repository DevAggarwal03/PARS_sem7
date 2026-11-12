"""Predict an unknown book rating with regularized matrix factorization and SGD."""
import numpy as np

from core import BOOK_IDS, BOOKS, MATRIX, RATINGS, READERS


def factorize(reader="R01", book_id="B03", epochs=20, factors=2,
              learning_rate=0.01, regularization=0.02):
    if reader not in READERS:
        raise ValueError("Choose a reader from the rating table.")
    if book_id not in BOOK_IDS:
        raise ValueError("Choose a book from the catalog.")
    if (reader, book_id) in RATINGS:
        raise ValueError("Choose a book without an existing rating for this reader.")
    if epochs < 3:
        raise ValueError("Use at least 3 training epochs.")
    if factors < 1 or learning_rate <= 0 or regularization < 0:
        raise ValueError("Factors and learning rate must be positive; regularization cannot be negative.")

    row_count, book_count = MATRIX.shape
    observed = [(u, i, MATRIX[u, i]) for u in range(row_count)
                for i in range(book_count) if not np.isnan(MATRIX[u, i])]
    mean_rating = float(np.mean([rating for _, _, rating in observed]))
    rng = np.random.default_rng(21)
    initial = np.sqrt(mean_rating / factors)
    reader_factors = rng.normal(initial, 0.05, size=(row_count, factors))
    book_factors = rng.normal(initial, 0.05, size=(book_count, factors))
    history = []

    for epoch in range(1, epochs + 1):
        rng.shuffle(observed)
        for u, i, actual in observed:
            reader_old = reader_factors[u].copy()
            book_old = book_factors[i].copy()
            error = actual - float(reader_old @ book_old)
            reader_factors[u] += learning_rate * (error * book_old - regularization * reader_old)
            book_factors[i] += learning_rate * (error * reader_old - regularization * book_old)

        squared_error = sum((actual - float(reader_factors[u] @ book_factors[i])) ** 2
                            for u, i, actual in observed)
        penalty = regularization * (float(np.sum(reader_factors ** 2))
                                    + float(np.sum(book_factors ** 2)))
        rmse = float(np.sqrt(squared_error / len(observed)))
        history.append({"epoch": epoch, "rmse": round(rmse, 4),
                        "squared_error": round(squared_error, 4),
                        "regularization_penalty": round(penalty, 4),
                        "loss": round(0.5 * squared_error + 0.5 * penalty, 4)})

    predictions = reader_factors @ book_factors.T
    user_index, book_index = READERS.index(reader), BOOK_IDS.index(book_id)
    predicted = float(predictions[user_index, book_index])
    return {
        "reader": reader,
        "book": next(book for book in BOOKS if book["id"] == book_id),
        "epochs": epochs,
        "factors": factors,
        "learning_rate": learning_rate,
        "regularization": regularization,
        "observed_count": len(observed),
        "mean_rating": round(mean_rating, 4),
        "predicted_rating": round(predicted, 4),
        "display_rating": round(float(np.clip(predicted, 1, 5)), 4),
        "was_clipped": predicted < 1 or predicted > 5,
        "history": history,
        "observed_matrix": [[None if np.isnan(value) else float(value) for value in row]
                            for row in MATRIX],
        "completed_matrix": [[round(float(value), 4) for value in row]
                             for row in predictions],
        "reader_factors": {name: [round(float(value), 4) for value in reader_factors[index]]
                           for index, name in enumerate(READERS)},
        "book_factors": {book["id"]: [round(float(value), 4) for value in book_factors[index]]
                         for index, book in enumerate(BOOKS)},
    }


def main():
    result = factorize()
    print(f"Predicted rating for {result['reader']} and {result['book']['title']}: "
          f"{result['display_rating']} / 5")
    print(f"Training observations: {result['observed_count']}; epochs: {result['epochs']}; "
          f"final RMSE: {result['history'][-1]['rmse']}")
    for epoch in result["history"]:
        print(f"Epoch {epoch['epoch']}: loss={epoch['loss']}, RMSE={epoch['rmse']}")


if __name__ == "__main__":
    main()
