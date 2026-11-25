# Week 2 – Machine Learning Assessment

## Overview

This repository contains the Week 2 Machine Learning assessment notebooks for the bank-loan prediction scenario.

The assessment scenario asks for a workflow that:
- Predicts whether a loan applicant will default (classification).
- Estimates the loan amount a customer may request (regression).
- Segments customers (clustering).

**Assessment duration:** About 6 hours  
**Mode:** Individual, hands-on

## Repository Structure

```text
Week2_ML_Assessment/
├── Task1_Concepts_Splits_Leakage.ipynb
├── Task2_Preprocessing_Pipeline.ipynb
├── Task3_Supervised_Unsupervised.ipynb
├── Task4_Evaluation_Tuning.ipynb
└── README.md
```

## Tasks Covered

### Task 1 – Concepts, Splits and Leakage
- Split data into training, validation, and test sets (60/20/20), using `random_state=42` and stratification.
- Keep the test set aside.
- Demonstrate overfitting with `DecisionTreeClassifier` at depths 1, 3, 5, 10, and `None`; plot training versus validation accuracy.
- Run 5-fold `cross_val_score` for Logistic Regression and report mean and standard deviation.
- Demonstrate incorrect scaling before splitting versus fitting preprocessing on training data only.
- Classify spam detection, customer segmentation, house price prediction, and anomaly detection as supervised or unsupervised.
- Describe the machine-learning workflow in seven steps.

### Task 2 – Preprocessing, Features and Pipeline
- Handle missing numeric and categorical values using `SimpleImputer` (median and most frequent, respectively).
- Try `KNNImputer` on numeric columns.
- Encode categorical features using `OneHotEncoder` and an ordered feature using `OrdinalEncoder`.
- Create at least three features: date decomposition, an income/loan interaction, and a log-transformed skewed feature.
- Build a `ColumnTransformer` inside a `Pipeline`.
- Save the fitted pipeline with `joblib`, reload it, and predict for three new raw rows.

### Task 3 – Models: Supervised and Unsupervised
- Train Logistic Regression, Decision Tree, Random Forest, and XGBoost classifiers (where XGBoost is installed).
- Print the Random Forest out-of-bag (OOB) score.
- Demonstrate `predict_proba` for five Logistic Regression rows and compare with `predict`.
- Display the five most important features from Random Forest and XGBoost (where available).
- Train Linear Regression, Ridge, and Lasso models to estimate loan amount; compare coefficients and count Lasso zero coefficients.
- Run K-Means for `k = 2` to `8`, plot the elbow curve, calculate silhouette scores, and summarize the segments.

### Task 4 – Evaluation and Tuning
- Report accuracy, precision, recall, F1 score, and confusion matrix for each classifier.
- Plot ROC curves for three models and report AUC.
- Select a decision threshold that achieves at least 75% recall and report its precision.
- Tune Random Forest using `GridSearchCV` with three parameters.
- Tune XGBoost using `RandomizedSearchCV` with `n_iter=15` (where XGBoost is installed).
- Report regression MAE, RMSE, and R-squared against a mean baseline.
- Evaluate the selected best classifier once on the held-out test set.

## How to Run

1. Download or clone this repository.
2. Open the folder in VS Code or Jupyter Notebook.
3. Select a Python 3 kernel.
4. Install the required packages:

   ```bash
   pip install numpy pandas matplotlib scikit-learn joblib jupyter
   ```

5. To run the XGBoost sections, install XGBoost:

   ```bash
   pip install xgboost
   ```

6. Open each `.ipynb` file and run its cells from top to bottom.

## Dataset Note

The assessment brief describes using a loan/customer dataset supplied by the trainer (for example, a credit or Telco-churn-style dataset with a binary target). The notebooks prepared for this repository use a **synthetic bank-loan dataset** because no trainer-provided dataset was included with the materials used to prepare them.

If your trainer has supplied a dataset, replace the dataset-generation section with the supplied data and update the feature/target column names as needed. Results from synthetic data demonstrate the workflow and should not be represented as results from the trainer's real dataset.

## Important Notes

- Work individually and use documentation where needed.
- Run and demonstrate every task.
- Commit changes regularly with clear Git commit messages.
- Do not commit `.env` files, passwords, API keys, or other secrets.
- Add short comments explaining important preprocessing and modelling decisions.

## Tools and Libraries

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- joblib
- XGBoost (optional, for XGBoost model sections)
- Jupyter Notebook

---

**Week 2 – Machine Learning Assessment**
