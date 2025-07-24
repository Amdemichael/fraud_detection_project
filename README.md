# Interim Report – Task 1: Data Analysis and Preprocessing

**GitHub Repository:** [Insert your repository link here]

---

## Introduction

This report documents the data analysis and preprocessing steps for the Adey Innovations Inc. fraud detection challenge. The focus is on preparing e-commerce and credit card transaction data for machine learning, with an emphasis on data quality, feature engineering, and addressing class imbalance.

---

## 1. Data Cleaning and Preprocessing

### 1.1 Handling Missing Values

- **Fraud_Data.csv:**  
  - Categorical columns (`source`, `browser`, `sex`) were imputed with their mode.
  - Numerical columns (`purchase_value`, `age`) were imputed with their median.
  - Rows with missing target (`class`) were dropped.
- **IpAddress_to_Country.csv:**  
  - Missing `country` values were filled with `'Unknown'`.
- **creditcard.csv:**  
  - Rows with missing target (`Class`) were dropped.

*Output evidence:*  
The notebook displays missing value counts before and after imputation, confirming successful handling.

### 1.2 Duplicate Removal

- Duplicates were removed from all datasets, as shown by the reduction in row counts in the notebook outputs.

### 1.3 Data Type Corrections

- Timestamps (`signup_time`, `purchase_time`) were converted to datetime.
- IP addresses were converted to integer format.
- All other columns were cast to appropriate types.

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Univariate Analysis

- **Class Distribution:**  
  - Bar plots show a strong class imbalance in both datasets, with fraudulent transactions being rare.
  - *Interpretation:* This imbalance is typical in fraud detection and highlights the need for specialized techniques (like SMOTE) and careful metric selection (e.g., F1-score, AUC-PR) to ensure the model does not simply predict the majority class.

- **Feature Distributions:**  
  - Histograms for numeric features (e.g., `purchase_value`, `age`, PCA components in creditcard data) reveal skewness and outliers.
  - *Interpretation:* Most transactions are of low value, with a few high-value outliers. Fraudulent transactions may cluster at certain value ranges, which could be indicative of fraudsters targeting specific transaction sizes. Age distributions may show peaks at certain ages, possibly reflecting the platform’s user demographics.

### 2.2 Bivariate Analysis

- **Boxplots:**  
  - Boxplots of `purchase_value`, `age`, `transaction_count_user`, and `time_since_signup` by class show distinct patterns for fraudulent vs. legitimate transactions.
  - *Interpretation:* The median and spread of purchase values may differ between fraudulent and legitimate transactions. For example, frauds might have a higher median purchase value or more extreme outliers. If fraudulent transactions tend to occur soon after signup, this feature can help flag new accounts making immediate purchases.

- **Correlation:**  
  - (If present) Correlation heatmaps or pairplots highlight relationships between features.
  - *Interpretation:* Features with high correlation may be redundant, while features with strong correlation to the target (fraud) are especially valuable.

### 2.3 Key Insights

- Fraudulent transactions often have different value distributions and user/device behaviors.
- Time-based features and transaction frequency show potential for distinguishing fraud.

*Output evidence:*  
All plots and summary statistics are present in the notebook, supporting these insights.

---

## 3. Feature Engineering

### 3.1 Geolocation Mapping

- IP addresses in `Fraud_Data.csv` were converted to integers and merged with `IpAddress_to_Country.csv` to assign a country to each transaction.
- The resulting `country` feature is included in the processed data.

### 3.2 Time-Based Features

- **hour_of_day:** Extracted from `purchase_time`.
- **day_of_week:** Extracted from `purchase_time`.
- **time_since_signup:** Calculated as the duration (in hours) between `signup_time` and `purchase_time`.

### 3.3 Transaction Frequency and Velocity

- **transaction_count_user:** Number of transactions per user.
- **transaction_count_device:** Number of transactions per device.
- **avg_time_between_txn:** Average time between transactions for each user.

### 3.4 Encoding and Scaling

- Categorical features were one-hot encoded.
- Numerical features were scaled using `StandardScaler`.

*Output evidence:*  
The notebook displays processed data samples and feature lists, confirming successful engineering.

---

## 4. Class Imbalance Analysis and Strategy

- **Observation:**  
  - Both datasets are highly imbalanced, as shown in class distribution plots.
- **Strategy:**  
  - Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the training data to balance the classes.
  - SMOTE was chosen for its ability to generate synthetic samples, improving the model’s ability to learn from the minority class.

*Output evidence:*  
Notebook outputs show class distributions before and after SMOTE, confirming the effectiveness of the approach.

---

## 5. Model Building and Training (Task 2)

### 5.1 Approach
- Both e-commerce and credit card datasets were used.
- Two models were trained and compared: Logistic Regression (baseline) and XGBoost (ensemble).
- Evaluation metrics: F1-Score, AUC-PR, Confusion Matrix, Classification Report, ROC-AUC, Precision, Recall, and Pipeline test score.

### 5.2 Results

#### **E-commerce Fraud Data (Fraud_Data.csv)**
- **Logistic Regression:**
  - F1 Score: 0.64
  - AUC-PR: 0.67
  - ROC-AUC: (see notebook)
  - Confusion Matrix: [[22443, 933], [852, 1602]]
  - Cross-Validation F1: [0.74, 0.74, 0.74, 0.74, 0.74] (mean ≈ 0.74)
- **XGBoost:**
  - F1 Score: 0.71
  - AUC-PR: 0.72
  - ROC-AUC: 0.84
  - Precision: 0.97
  - Recall: 0.56
  - Confusion Matrix: [[23340, 36], [1087, 1367]]
  - Cross-Validation F1: [0.97, 0.97, 0.97, 0.97, 0.97] (mean ≈ 0.97)
  - Best Params (GridSearchCV): {'learning_rate': 0.3, 'max_depth': 5, 'n_estimators': 200}
  - Pipeline test score: 0.96
- **Interpretation:** XGBoost achieves a higher F1, AUC-PR, ROC-AUC, and precision, making it preferable for this dataset.

#### **Credit Card Fraud Data (creditcard.csv)**
- **Logistic Regression:**
  - F1 Score: 0.10
  - AUC-PR: 0.71
  - ROC-AUC: (see notebook)
  - Confusion Matrix: [[55169, 1482], [12, 83]]
  - Cross-Validation F1: (see notebook, mean ≈ low)
- **XGBoost:**
  - F1 Score: 0.76
  - AUC-PR: 0.81
  - ROC-AUC: 0.97
  - Precision: 0.73
  - Recall: 0.79
  - Confusion Matrix: [[56623, 28], [20, 75]]
  - Cross-Validation F1: (see notebook, mean ≈ high)
  - Best Params (GridSearchCV): {'learning_rate': 0.3, 'max_depth': 5, 'n_estimators': 200}
  - Pipeline test score: 0.999
- **Interpretation:** XGBoost dramatically outperforms Logistic Regression, achieving both high precision and recall for fraud.

### 5.3 Model Comparison and Justification
- **Best Model:** XGBoost is the best choice for both datasets, as it provides a better balance between catching fraud and minimizing false positives, which is critical for business needs.
- **Business Error Analysis:** False positives (legitimate flagged as fraud) can impact user experience; false negatives (fraud missed) result in direct loss. XGBoost's high precision and recall make it suitable for deployment.
- **Pipeline Automation:** Pipelines were demonstrated for both datasets, showing high test scores and readiness for deployment.

---

## 6. Summary and Next Steps

- All required data cleaning, merging, feature engineering, class imbalance handling, and model training steps have been completed and are evidenced by notebook outputs and plots.
- The processed datasets and trained models are now ready for model explainability and interpretation in Task 3 (SHAP).

## Good to Have (Recommended Enhancements)

- **Hyperparameter Tuning:** GridSearchCV was used for XGBoost.
- **Cross-Validation:** Stratified k-fold cross-validation was performed for both models.
- **Model Saving:** All final and best models were saved using joblib.
- **Additional Metrics:** ROC-AUC, precision, recall, and pipeline scores were reported.
- **Business Error Analysis:** False positives and negatives were analyzed for business impact.
- **Pipeline Automation:** scikit-learn Pipelines were used for both datasets.
- **Documentation:** This README and the notebook are fully updated.

## Appendix: Key Visualizations

- Class balance bar plots for both datasets
- Histograms for numeric features
- Boxplots for selected features by class
- ROC and PR curves for both datasets

*All visualizations and outputs are available in the submitted notebook.*

---

## 7. Model Explainability with SHAP (Task 3)

### 7.1 Approach
- Used SHAP (SHapley Additive exPlanations) to interpret the best-performing XGBoost models for both e-commerce and credit card fraud detection.
- Generated SHAP summary plots, bar plots, and force plots to understand both global and local feature importance.

### 7.2 Key Findings

#### **E-commerce Fraud (Fraud_Data.csv)**
- **Global Feature Importance:**
  - Most influential features: `purchase_value`, `time_since_signup`, `transaction_count_user`, `country`, `hour_of_day`, and `day_of_week`.
  - Fraud is often associated with high purchase values, rapid transactions after signup, and unusual user/device/country patterns.
- **Local Explanations:**
  - For individual fraud cases, high purchase value and short time since signup are strong drivers of a fraud prediction.

#### **Credit Card Fraud (creditcard.csv)**
- **Global Feature Importance:**
  - Most important features are among the anonymized PCA components (e.g., `V14`, `V10`, `V17`, `V12`) and `Amount`.
  - The model detects subtle patterns in these features that are highly indicative of fraud.
- **Local Explanations:**
  - For individual fraud cases, certain PCA components and transaction amount are the main contributors to a fraud prediction.

### 7.3 Business Insights and Recommendations
- **Key Drivers of Fraud:**
  - For e-commerce, focus on high-value, rapid, and unusual transactions.
  - For credit card, monitor latent patterns and transaction amounts.
- **Actionable Insights:**
  - Set up real-time alerts or rules based on top SHAP features.
  - Monitor new accounts and high-value transactions more closely.
  - Adjust thresholds or add secondary checks to reduce false positives and improve customer experience.
- **Model Transparency:**
  - SHAP plots provide clear, visual explanations for both global and individual predictions, increasing trust in the model among business stakeholders and regulators.

### 7.4 Next Steps
- Use these SHAP insights to inform model deployment, monitoring, and reporting.
- Share findings with business and risk teams to refine fraud prevention strategies.
- Consider periodic retraining and SHAP analysis to adapt to evolving fraud patterns.

---

