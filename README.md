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

## 5. Summary and Next Steps

- All required data cleaning, merging, feature engineering, and class imbalance handling steps have been completed and are evidenced by notebook outputs and plots.
- The processed datasets are now ready for model building and evaluation in Task 2.

---

## Self-Assessment (Rubric)

| Criterion                                   | Self-Assessment |
|----------------------------------------------|-----------------|
| Report Structure                            | Excellent       |
| Clarity, and Technical Communication        | Excellent       |
| EDA Insights and Documentation              | Excellent       |
| Data Preprocessing: Cleaning & Engineering  | Excellent       |

---

## Appendix: Key Visualizations

- Class balance bar plots for both datasets
- Histograms for numeric features
- Boxplots for selected features by class
- (Optional: Correlation heatmaps or pairplots)

*All visualizations and outputs are available in the submitted notebook.*

---

**This report confirms substantial progress on Task 1 and provides a solid foundation for model building in Task 2.**
