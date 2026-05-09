# 📡 Customer Churn Prediction App

A production-ready machine learning project that predicts telecom customer churn using an end-to-end sklearn Pipeline, with a Streamlit web app for single-customer and batch CSV predictions.

---

## ✨ Features

- **Dual prediction modes** — manual single-customer form & batch CSV upload
- **Full sklearn Pipeline** — StandardScaler + OneHotEncoder + RandomForest/GradientBoosting/LogisticRegression
- **Auto column validation** — detects missing columns and suggests fixes
- **Download predictions** — export results as `churn_predictions.csv`
- **Rich visualizations** — churn distribution, probability histogram, risk segmentation, feature importances
- **Model comparison** — trains 3 models, selects best by F1 Score
- **Sample CSV template** — downloadable from sidebar

---

## 🗂 Project Structure

```
churn_project/
├── app/
│   └── app.py              # Streamlit web application
├── data/
│   ├── churn_train.csv     # Generated after training
│   └── sample_template.csv # Sample CSV for upload testing
├── models/
│   ├── churn_pipeline.pkl  # Saved sklearn Pipeline
│   └── training_metrics.json
├── utils/
│   ├── data_generator.py   # Synthetic dataset generator
│   ├── model_utils.py      # Training & evaluation helpers
│   ├── preprocessing.py    # Validation & cleaning helpers
│   └── visualizations.py   # Matplotlib/Seaborn plot functions
├── train.py                # Run this first to train the model
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
This generates the dataset, trains 3 models, selects the best, and saves `models/churn_pipeline.pkl`.

### 3. Launch the app
```bash
streamlit run app/app.py
```

---

## 📋 Required CSV Columns

| Column | Type | Example |
|---|---|---|
| tenure | int | 24 |
| MonthlyCharges | float | 65.50 |
| TotalCharges | float | 1572.00 |
| SeniorCitizen | int (0/1) | 0 |
| gender | str | Male |
| Partner | str | Yes |
| Dependents | str | No |
| PhoneService | str | Yes |
| MultipleLines | str | No |
| InternetService | str | Fiber optic |
| OnlineSecurity | str | No |
| OnlineBackup | str | Yes |
| DeviceProtection | str | No |
| TechSupport | str | No |
| StreamingTV | str | Yes |
| StreamingMovies | str | No |
| Contract | str | Month-to-month |
| PaperlessBilling | str | Yes |
| PaymentMethod | str | Electronic check |

---

## 🛠 Tech Stack

- Python 3.10+
- Scikit-learn (Pipeline, ColumnTransformer, RandomForest, GradientBoosting, LogisticRegression)
- Streamlit
- Pandas / NumPy
- Matplotlib / Seaborn
