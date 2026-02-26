"""
Linear & Logistic Regression Lab

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 everywhere required.
"""

import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():

    # STEP 1: Load dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # STEP 2: Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: Standardize features (fit only on train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4: Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # STEP 5: Compute metrics
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # STEP 6: Top 3 features by absolute coefficient value
    coef_abs = np.abs(model.coef_)
    top_3_feature_indices = list(np.argsort(coef_abs)[-3:][::-1])

    # Does the model overfit?
    # If train R² is much higher than test R², model may be overfitting.
    # If they are close, overfitting is minimal.

    # Why is feature scaling important?
    # Scaling ensures all features are on the same scale,
    # which stabilizes optimization and makes coefficients comparable.

    return train_mse, test_mse, train_r2, test_r2, top_3_feature_indices


# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():

    # STEP 1: Load dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # STEP 2: Standardize entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # STEP 3: 5-fold Cross-validation
    model = LinearRegression()
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

    # STEP 4: Compute mean and std
    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)

    # Standard deviation represents how much model performance
    # varies across different folds.

    # Cross-validation reduces variance risk by averaging performance
    # across multiple train/test splits instead of relying on one split.

    return mean_r2, std_r2


# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():

    # STEP 1: Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # STEP 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4: Train Logistic Regression
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # STEP 5: Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred)

    # False Negative (medical meaning):
    # A False Negative means the model predicts "no cancer"
    # when the patient actually HAS cancer.
    # This is dangerous because treatment may be delayed.

    return train_accuracy, test_accuracy, precision, recall, f1


# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():

    # STEP 1: Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # STEP 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # STEP 4: Train for different C values
    for C_value in [0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(max_iter=5000, C=C_value)
        model.fit(X_train_scaled, y_train)

        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        results[C_value] = (train_acc, test_acc)

    # When C is very small:
    # Strong regularization → simpler model → may underfit.

    # When C is very large:
    # Weak regularization → complex model → risk of overfitting.

    # Overfitting usually occurs at very large C values.

    return results


# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():

    # STEP 1: Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # STEP 2: Standardize entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # STEP 3: 5-fold CV
    model = LogisticRegression(C=1, max_iter=5000)
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

    # STEP 4: Mean and Std
    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)

    # Cross-validation is critical in medical diagnosis because
    # it ensures the model performs consistently across different
    # patient subsets and is not biased toward one specific split.

    return mean_accuracy, std_accuracy
