# Dataset Study Webpage (Flask)

A single-page Flask website presenting two datasets from different domains
(UCI Heart Disease — healthcare, and CIFAR-10 — computer vision), including
purpose, source, feature tables, author performance comparison tables, and
references.

## How to Run

1. (Optional) Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   python app.py
   ```

4. Open your browser at:
   ```
   http://127.0.0.1:5000/
   ```

## Project Structure

```
dataset_assignment/
├── app.py                 # Flask entry point
├── requirements.txt
├── templates/
│   └── index.html         # Single page with both dataset sections
└── static/
    └── style.css           # Simple, formal styling
```
