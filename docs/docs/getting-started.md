# Getting Started

This guide will help you set up the 12FactorApp project on a clean install.

---

## 1. Clone the Repository

```sh
git clone https://github.com/yourusername/12factorapp.git
cd 12factorapp
```

## 2. Set Up a Virtual Environment

```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 3. Install Dependencies

```sh
pip install -r requirements.txt
```

## 4. Prepare the Data

- Place your raw data files in the `data/raw/` directory.
- Use the provided scripts or notebooks to process and explore the data.

## 5. Run the Notebooks

- Open `notebooks/house_prediction.ipynb` in Jupyter or VS Code for exploratory analysis and prototyping.

## 6. Train the Model

```sh
python 12factorapp/modeling/train.py
```

## 7. Serve the API

```sh
uvicorn 12factorapp.api:app --reload
```
- Access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

## 8. Build and Serve Documentation

```sh
mkdocs build
mkdocs serve
```

---

For more details, see the [Project Overview](index.md).
