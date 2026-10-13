# Customer Churn Prediction

Predicts which telecom customers are likely to leave, using the Telco Customer Churn dataset ([add the Kaggle link]).

## Problem
Keeping a customer is cheaper than winning a new one. The goal is to flag customers likely to churn so the company can contact them early.

## Data
7043 customers, 26.5% churned (imbalanced).

## What I did
- Dropped leakage columns: Churn Reason, Churn Score, CLTV
- Dropped single-value, ID and high-cardinality columns
- Fixed Total Charges (text with blanks for tenure-0 customers, filled with 0)
- Encoded categorical columns; removed 6 duplicate "No internet service" columns
- Stratified 80/20 split; scaling inside a Pipeline to avoid leakage
- Tuned with 5-fold cross-validation

## Results (test set)
| Model | Accuracy | ROC-AUC | Recall (churn) | Precision (churn) |
|---|---|---|---|---|
| Always predict "stays" | 73.5% | 0.50 | 0 | n/a |
| Logistic Regression (default) | 80.1% | 0.849 | 0.57 | 0.64 |
| Decision Tree | 72.1% | 0.640 | 0.47 | 0.47 |
| Random Forest | 79.8% | 0.837 | 0.52 | 0.65 |
| Tuned Logistic Regression (balanced, threshold 0.4) | 70.8% | 0.846 | 0.86 | 0.47 |

## Key findings
- Longer tenure, two-year contracts and having dependents go with staying; fiber optic, higher monthly charges and electronic check go with churning
- [One thing that surprised you, in your own words]

## Lessons
- Accuracy is misleading on imbalanced data
- An unrestricted decision tree overfit (99.8% train vs 72% test)
- The threshold is a business decision, not a model property
- [One lesson in your own words]

## Limitations
- The model misses churners who look loyal; it would need features like support calls or usage trends
- The demo app varies only 4 inputs; other features are held at typical values

## Run the demo
pip install -r requirements.txt
streamlit run app.py