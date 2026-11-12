import json
from pathlib import Path

import exp_12, exp_13, exp_14, exp_15, exp_16, exp_17, exp_18, exp_19, exp_20, exp_21

ROOT = Path(__file__).resolve().parent
TASKS = {
    12: ("Content-based recommendations", "Ranks books by overlap between genres/authors of well-rated books and catalog features.", exp_12.recommend, {"reader": "R01"}),
    13: ("Collaborative filtering", "Uses rating votes from readers with positive centered similarity.", exp_13.recommend, {"reader": "R01"}),
    14: ("Cold-start recommendations", "Uses a selected genre and catalog popularity without requiring reader history.", exp_14.recommend, {"reader": None, "genres": ["Mystery"]}),
    15: ("User-based nearest neighbors", "Adds a similarity-weighted average of neighbor rating deviations to the selected reader's mean rating.", exp_15.recommend, {"reader": "R01", "neighbors": 3}),
    16: ("Item-based nearest neighbors", "Predicts ratings from similar books in the reader's history.", exp_16.recommend, {"reader": "R01"}),
    17: ("Adjusted cosine similarity", "Centers ratings by reader before measuring similarity between books.", exp_17.recommend, {"reader": "R01"}),
    18: ("Item-based k-nearest neighbors", "Predicts from the closest rated books for either read or unread catalog items.", exp_18.recommend, {"reader": "R01", "status": "unseen", "neighbors": 3}),
    19: ("Hybrid recommendations", "Combines content match, collaborative predictions, and popularity.", exp_19.recommend, {"reader": "R01"}),
    20: ("Knowledge-based recommendations", "Scores genre, author, and page-range constraints with fixed weights.", exp_20.recommend, {"genre": "Mystery", "author": "", "min_pages": 0, "max_pages": 500}),
    21: ("Matrix factorization with SGD", "Learns reader and book latent factors with regularized stochastic gradient descent to estimate an unknown rating.", exp_21.factorize, {"reader": "R01", "book_id": "B03", "epochs": 20}),
}


def main():
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    for number, (title, method, function, inputs) in TASKS.items():
        output = function(**inputs)
        terminal = repr(output)
        record = {
            "assignment": number,
            "title": title,
            "libraries": ["Python", "Flask", "NumPy"],
            "methodology": method,
            "related_files": [f"exp_{number}.py", "app.py", "core.py", "catalog.json", "ratings.csv"],
            "inputs": inputs,
            "results": output,
            "terminal_output": terminal,
        }
        destination = output_dir / f"exp_{number}.json"
        destination.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary = (f"rating {output['display_rating']} after {output['epochs']} epochs"
                   if number == 21 else f"{len(output)} ranked books")
        print(f"Assignment {number}: {summary} -> results/exp_{number}.json")


if __name__ == "__main__":
    main()
