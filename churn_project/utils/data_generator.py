"""
Synthetic Telecom Churn Dataset Generator
Generates realistic customer data mimicking real-world telecom patterns.
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_telecom_churn_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic telecom churn dataset.

    Features are correlated with churn probability to reflect real-world patterns:
    - Higher monthly charges → higher churn
    - Longer tenure → lower churn
    - Month-to-month contracts → higher churn
    - No tech support → higher churn

    Args:
        n_samples: Number of customer records to generate.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with customer features and Churn label.
    """
    rng = np.random.default_rng(random_state)

    # --- Demographics ---
    gender = rng.choice(["Male", "Female"], size=n_samples)
    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])

    # --- Service Info ---
    tenure = rng.integers(0, 73, size=n_samples)  # months 0–72
    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n_samples, p=[0.50, 0.50]),
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22]
    )

    def internet_feature(yes_prob=0.5):
        return np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n_samples, p=[yes_prob, 1 - yes_prob]),
        )

    online_security = internet_feature(0.29)
    online_backup = internet_feature(0.34)
    device_protection = internet_feature(0.34)
    tech_support = internet_feature(0.29)
    streaming_tv = internet_feature(0.38)
    streaming_movies = internet_feature(0.39)

    # --- Contract & Billing ---
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.55, 0.21, 0.24],
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n_samples,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # Monthly charges: fiber optic customers pay more
    base_charge = np.where(
        internet_service == "Fiber optic",
        rng.uniform(70, 120, n_samples),
        np.where(
            internet_service == "DSL",
            rng.uniform(45, 85, n_samples),
            rng.uniform(20, 55, n_samples),
        ),
    )
    monthly_charges = np.round(base_charge, 2)

    # Total charges ≈ tenure × monthly_charges + noise
    total_charges = np.round(
        tenure * monthly_charges + rng.uniform(-50, 50, n_samples), 2
    )
    total_charges = np.maximum(total_charges, 0)

    # --- Churn Label (logistic model) ---
    churn_score = (
        -2.5
        + 0.03 * (monthly_charges - 60)          # higher bill → more churn
        - 0.05 * tenure                            # longer tenure → less churn
        + 1.5 * (contract == "Month-to-month").astype(int)
        - 1.0 * (contract == "Two year").astype(int)
        + 0.8 * (internet_service == "Fiber optic").astype(int)
        + 0.6 * (tech_support == "No").astype(int)
        + 0.5 * (online_security == "No").astype(int)
        + 0.4 * (payment_method == "Electronic check").astype(int)
        + 0.3 * senior_citizen
        + rng.normal(0, 0.5, n_samples)            # noise
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    churn = np.where(rng.random(n_samples) < churn_prob, "Yes", "No")

    df = pd.DataFrame(
        {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )
    return df


def get_feature_columns():
    """Return ordered lists of numerical and categorical feature columns (no target)."""
    numerical = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    return numerical, categorical


def get_all_feature_columns():
    """Return all feature columns (numerical + categorical)."""
    num, cat = get_feature_columns()
    return num + cat


if __name__ == "__main__":
    df = generate_telecom_churn_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
