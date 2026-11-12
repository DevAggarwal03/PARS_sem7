# Extended Book Recommendation Assignments

The Flask application demonstrates assignments 12–21 using a local book catalog and reader rating table.

## Run

From this folder:

```bash
uv sync
uv run python run_assignments.py
uv run python app.py
```

Open `http://127.0.0.1:5002/` to use the pages.

Run one recommender directly with `uv run python exp_12.py` through `uv run python exp_20.py` to print ranked books, or run `uv run python exp_21.py` to print a matrix-factorization prediction and epoch progress. Run `uv run python run_assignments.py` to refresh all ten JSON result records. The UI and dataset work offline.

## Project contents

- `catalog.json` stores book details; `ratings.csv` stores reader ratings.
- `exp_12.py` through `exp_20.py` each implement one method.
- `exp_21.py` predicts missing reader-book ratings with regularized matrix factorization and SGD.
- `app.py`, `templates/`, and `static/` provide the shared Flask interface.
- `results/exp_12.json` through `results/exp_21.json` record method, inputs, file references, output, and terminal text.
- `plot_results/` is reserved for plots if the assignment requirements call for them.
