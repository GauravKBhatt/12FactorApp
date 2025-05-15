# 12FactorApp

[![CCDS Project template](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

A simple, modular FastAPI application for house price prediction, following the 12-factor app methodology and best practices for reproducible data science projects.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project demonstrates a production-ready, modular data science workflow using FastAPI for serving a house price prediction model. It is structured using the [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/) template and follows the 12-factor app principles for maintainability and scalability.

---

## Features

- **FastAPI**: REST API for model inference.
- **Modular Codebase**: Separation of data, features, modeling, and visualization.
- **Reproducibility**: Environment and dependency management.
- **Notebooks**: For exploratory data analysis and prototyping.
- **Automated Testing**: With pytest.
- **Documentation**: Built with MkDocs.

---

## Project Structure

```
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw
├── docs
│   └── docs
│       ├── index.md
│       └── getting-started.md
├── models
├── notebooks
│   └── house_prediction.ipynb
├── pyproject.toml
├── references
├── reports
│   └── figures
├── requirements.txt
├── setup.cfg
├── mkdocs.yml
└── 12factorapp
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── features.py
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py
    │   └── train.py
    └── plots.py
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/12factorapp.git
    cd 12factorapp
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **(Optional) Install development dependencies:**
    ```sh
    pip install -r requirements-dev.txt
    ```

---

## Usage

### Data Preparation

- Place your raw data in `data/raw/`.
- Use the provided scripts or notebooks to process and explore data.

### Model Training

- Run the training script:
    ```sh
    python 12factorapp/modeling/train.py
    ```

### Model Inference (API)

- Start the FastAPI server:
    ```sh
    uvicorn 12factorapp.api:app --reload
    ```
- Access the API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### Notebooks

- Explore and prototype in `notebooks/house_prediction.ipynb`.

---

## Development

### Code Formatting & Linting

- Format and lint code using [ruff](https://github.com/astral-sh/ruff):
    ```sh
    make format
    make lint
    ```

### Cleaning

- Remove Python cache and compiled files:
    ```sh
    make clean
    ```

---

## Testing

- Run all tests:
    ```sh
    make test
    ```

---

## Documentation

- Build documentation locally:
    ```sh
    mkdocs build
    ```
- Serve documentation locally:
    ```sh
    mkdocs serve
    ```
- Documentation source: `docs/docs/`

---

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes.
4. Push to your fork and submit a pull request.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
