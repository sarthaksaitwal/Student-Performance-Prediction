# Student Performance Prediction

This project aims to predict student performance based on various features using machine learning techniques. The application provides a web interface for users to upload data, train models, and make predictions.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **Student Performance Prediction** project is designed to:
- Ingest and preprocess student data.
- Train machine learning models to predict performance.
- Provide a user-friendly web interface for predictions.

---

## Features

- **Data Ingestion**: Load and preprocess raw data.
- **Data Transformation**: Handle missing values, feature scaling, and encoding.
- **Model Training**: Train and evaluate machine learning models.
- **Prediction Pipeline**: Predict student performance using trained models.
- **Web Interface**: Upload data and view predictions via a web application.

---

## Project Structure

```
Student-Performance-Prediction/
│
├── app.py                # Main application file
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
│
├── artifacts/            # Stores raw, train, and test datasets
│   ├── raw.csv
│   ├── train.csv
│   └── test.csv
│
├── data/                 # Additional data files
│   └── stud.csv
│
├── logs/                 # Log files for debugging
│
├── src/                  # Source code
│   ├── components/       # Data ingestion, transformation, and model training
│   ├── pipeline/         # Training and prediction pipelines
│   ├── exception.py      # Custom exception handling
│   ├── logger.py         # Logging setup
│   └── utils.py          # Utility functions
│
├── static/               # Static files (CSS, JS, images)
│   └── css/
│       └── style.css
│
└── templates/            # HTML templates for the web interface
    └── index.html
```

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Student-Performance-Prediction.git
   cd Student-Performance-Prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Run the application**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   Open your browser and navigate to `http://127.0.0.1:5000`.

3. **Upload data and make predictions**:
   - Use the web interface to upload a CSV file.
   - View predictions and model performance.

---

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS
- **Machine Learning**: scikit-learn, pandas, numpy
- **Logging**: Python `logging` module

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

