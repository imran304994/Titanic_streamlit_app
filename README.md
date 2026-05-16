# Titanic Survival Prediction App

A supervised binary classification machine learning project and web application that predicts passenger survival probability during the historic Titanic shipwreck based on demographic, socio-economic, and travel data. 

## 📊 Dataset Profile & Features

The model analyzes key passenger attributes from the historic manifest to establish survival correlations:
* **Pclass:** Socio-economic status proxy (1st, 2nd, or 3rd class).
* **Sex & Age:** Demographic parameters capturing historic emergency evacuation protocol priority ("women and children first").
* **SibSp & Parch:** Family structure attributes tracking numbers of siblings, spouses, parents, or children aboard.
* **Fare & Ticket:** Financial variables showing passenger accommodation categories.
* **Embarked:** Port of boarding (C = Cherbourg, Q = Queenstown, S = Southampton).

The target label (`Survived`) maps binary outcomes: **0 = Deceased**, **1 = Survived**.

## 🚀 Key Features

* **Data Handling & Pipeline Optimization:** Automatically handles structured training and evaluation splitting using separate `Titanic_train.csv` and `Titanic_test.csv` distributions.
* **Statistical Feature Parsing:** Explores missing value patterns across sparse structural variables (like `Age` and `Cabin`).
* **Supervised Binary Classification:** Implements a highly interpretable **Logistic Regression** model optimized for calculating clear survival log-odds coefficients.
* **Serialized Model Deployment:** Packages data weights into a production-ready serialization pipeline using Python `pickle` (`titanic.pkl`).
* **Interactive Frontend UI:** Serves predictions live via a responsive, client-facing dashboard framework constructed in **Streamlit** (`app.py`).

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning Library:** Scikit-Learn (Logistic Regression)
* **Model Serialization:** Pickle
* **Web Deployment Framework:** Streamlit

## 📁 Project Structure

```text
├── data/
│   ├── Titanic_train.csv     # Historical passenger training data
│   └── Titanic_test.csv      # Test verification dataset
├── app.py                    # Streamlit frontend application script
├── titanic.pkl               # Serialized trained Logistic Regression model
├── train_model.ipynb         # Jupyter Notebook detailing EDA, cleaning, and ML training
└── requirements.txt          # Library dependencies block
