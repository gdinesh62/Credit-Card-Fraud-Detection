# Credit Card Fraud Detection

An end-to-end machine learning project for detecting fraudulent credit-card transactions in a highly imbalanced dataset. The project compares a class-weighted Logistic Regression baseline with a SMOTE + Random Forest pipeline and selects the better model using fraud-focused evaluation metrics.

## Problem statement

Credit-card fraud is rare but costly. The dataset contains approximately **284,000 legitimate transactions** and **492 fraudulent transactions**, so accuracy is not a meaningful primary metric. This project focuses on identifying fraud while controlling unnecessary alerts on legitimate transactions.

## Approach

1. Loaded, inspected, and checked data quality for the transaction dataset.
2. Created stratified train, validation, and test splits to preserve the fraud rate in each split.
3. Built a class-weighted Logistic Regression model as a baseline.
4. Built a Random Forest pipeline with median imputation, scaling, and SMOTE applied only to training folds.
5. Evaluated both models using PR-AUC, ROC-AUC, precision, recall, F1-score, and the confusion matrix.
6. Selected the **SMOTE + Random Forest** model and tuned its prediction threshold using validation data.
7. Evaluated the chosen model once on the untouched test dataset, then saved it for reuse.

## Results

The SMOTE + Random Forest model was selected because it performed better than the Logistic Regression baseline on validation data, especially for fraud-focused metrics.

| Metric | Final test result |
| --- | ---: |
| PR-AUC | `ADD_VALUE` |
| ROC-AUC | `ADD_VALUE` |
| Fraud recall | `ADD_VALUE` |
| Fraud precision | `ADD_VALUE` |
| F1-score | `ADD_VALUE` |
| Probability threshold | `ADD_VALUE` |

Replace `ADD_VALUE` with the values printed by the final test-evaluation cell in the notebook. Avoid reporting accuracy alone because the dataset is severely imbalanced.

## Project structure

```text
credit-card-fraud-detection/
├── credit_card_fraud_detection.ipynb
├── creditcard.csv                    # Download separately; do not commit if restricted/large
├── fraud_detection_smote_model.joblib
├── fraud_detection_config.joblib
├── fraud_test_predictions.csv
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

Create a `requirements.txt` file containing:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
jupyter
```

## Running the project

1. Place the CSV dataset in the project root as `creditcard.csv`.
2. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open `credit_card_fraud_detection.ipynb` and run cells in order.
4. The notebook saves the trained model as `fraud_detection_smote_model.joblib` and the selected threshold/configuration as `fraud_detection_config.joblib`.

## Making predictions

The saved model returns a fraud probability. A transaction is flagged when its probability is at or above the validation-selected threshold.

```python
import joblib

model = joblib.load("fraud_detection_smote_model.joblib")
config = joblib.load("fraud_detection_config.joblib")

# new_transactions must have the same feature columns as the training data
fraud_probability = model.predict_proba(new_transactions)[:, 1]
fraud_prediction = (fraud_probability >= config["threshold"]).astype(int)
```

## Important considerations

- SMOTE is applied only after the train/validation/test split to prevent data leakage.
- The classification threshold should reflect the cost of missed fraud, false alerts, and the capacity of a manual review team.
- Fraud behavior changes over time; monitor recall, precision, fraud rate, and data drift after deployment.
- For real production systems, prefer time-based validation where transaction timestamps are available.

## Resume summary

- Built an end-to-end credit-card fraud detection pipeline for a dataset of approximately 284K legitimate and 492 fraudulent transactions.
- Compared Logistic Regression against SMOTE + Random Forest using PR-AUC, recall, precision, and cost-aware error analysis.
- Deployed a reusable trained-model pipeline with threshold-based fraud scoring for transaction review decisions.

## Dataset

Use the dataset in accordance with its license and terms of use. Do not upload sensitive transaction data, personal information, or files that cannot legally be redistributed to a public GitHub repository.

## License

This project is available under the MIT License. Add a `LICENSE` file if you would like to publish it under that license.
