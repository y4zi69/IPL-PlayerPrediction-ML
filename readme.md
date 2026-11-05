# 🏏 IPL Player Performance Predictor

## Overview
It is live on https://iplplayerperfomanceprediction.streamlit.app/#match-info
The **IPL Player Performance Predictor** is a Machine Learning web application built with Python and Streamlit. It predicts the number of runs a cricket player is expected to score in an Indian Premier League (IPL) match based on historical data, match conditions, venue, and the player's recent form.

## Features

* **Interactive Web Interface:** A clean, user-friendly UI built with Streamlit.

* **Real-Time Predictions:** Uses a pre-trained Machine Learning model (`best_model.pkl`) to generate instant run projections.

* **Dynamic Calculations:** Automatically calculates the player's recent strike rate based on runs and balls faced.

* **Comprehensive Match Context:** Accounts for factors like venue, pitch behavior, opponent bowling rank, and home advantage.

## Tech Stack

* **Language:** Python 3.10

* **Data Processing:** Pandas, NumPy

* **Machine Learning:** Scikit-Learn, XGBoost, Joblib

* **Web Framework:** Streamlit

* **Environment Management:** Conda

## Installation & Setup

1. **Clone the repository** (or download the project folder):

   ```bash
   git clone https://github.com/y4zi69/ipl-prediction.git
   cd ipl-prediction
   ```

2. **Create and activate a virtual environment:**

   ```bash
   conda create -n ipl_ml python=3.10 -y
   conda activate ipl_ml
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   conda install pandas numpy scikit-learn matplotlib seaborn jupyter -y
   pip install xgboost joblib streamlit
   ```

4. **Ensure the model exists:**
   Make sure your trained model file (`best_model.pkl`) is located inside the `models/` folder. If you need to train it from scratch, run your training script first (e.g., `python src/train_models.py`).

## Usage

To start the web application, ensure your `ipl_ml` environment is activated, then run:

```bash
streamlit run app.py
```

Your default web browser will automatically open the app at `http://localhost:8501`.

## Copyright

**Copyright (c) 2026 Mohammed Yasin N (https://github.com/y4zi69)**

All rights reserved. This project and its contents may not be copied, modified, or distributed without express permission.
