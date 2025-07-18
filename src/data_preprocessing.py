import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os
from typing import Tuple, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def handle_missing_values(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Handle missing values for different datasets."""
    logging.info(f"Checking missing values in {dataset_name}")
    print(f"\nMissing Values in {dataset_name}:\n", df.isnull().sum())
    try:
        if dataset_name == 'fraud_data':
            for col in ['source', 'browser', 'sex']:
                if col in df.columns:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    logging.info(f"Imputed {col} with mode: {df[col].mode()[0]}")
            for col in ['purchase_value', 'age']:
                if col in df.columns:
                    df[col].fillna(df[col].median(), inplace=True)
                    logging.info(f"Imputed {col} with median: {df[col].median()}")
            if 'class' in df.columns:
                df.dropna(subset=['class'], inplace=True)
                logging.info(f"Dropped rows with missing 'class'. New shape: {df.shape}")
        elif dataset_name == 'ip_country':
            if 'country' in df.columns:
                df['country'].fillna('Unknown', inplace=True)
                logging.info("Imputed missing 'country' with 'Unknown'")
        elif dataset_name == 'creditcard':
            if 'Class' in df.columns:
                df.dropna(subset=['Class'], inplace=True)
                logging.info(f"Dropped rows with missing 'Class'. New shape: {df.shape}")
    except Exception as e:
        logging.error(f"Error handling missing values: {e}")
    return df

def clean_data(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Remove duplicates and correct data types."""
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    logging.info(f"Removed {initial_rows - len(df)} duplicates from {dataset_name}. New shape: {df.shape}")
    try:
        if dataset_name == 'fraud_data':
            if 'signup_time' in df.columns:
                df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
            if 'purchase_time' in df.columns:
                df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
            if 'ip_address' in df.columns:
                # Convert to int if not already
                if df['ip_address'].dtype == object:
                    df['ip_address'] = df['ip_address'].apply(lambda x: int(x) if pd.notnull(x) and str(x).isdigit() else pd.NA)
                df['ip_address'] = pd.to_numeric(df['ip_address'], errors='coerce').astype('Int64')
            logging.info("Converted signup_time, purchase_time to datetime, ip_address to int")
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
    return df

def merge_ip_country(fraud_data: pd.DataFrame, ip_country: pd.DataFrame) -> pd.DataFrame:
    """Efficiently map IP addresses to countries using merge_asof."""
    try:
        # Ensure IP columns are sorted and integer
        ip_country = ip_country.sort_values('lower_bound_ip_address')
        fraud_data = fraud_data.sort_values('ip_address')
        merged = pd.merge_asof(
            fraud_data,
            ip_country,
            left_on='ip_address',
            right_on='lower_bound_ip_address',
            direction='backward',
            suffixes=('', '_country')
        )
        # Filter where ip_address <= upper_bound_ip_address
        merged = merged[merged['ip_address'] <= merged['upper_bound_ip_address']]
        merged['country'] = merged['country'].fillna('Unknown')
        logging.info(f"Merged IP to Country. Country Distribution:\n{merged['country'].value_counts()}")
        return merged
    except Exception as e:
        logging.error(f"Error merging IP to country: {e}")
        fraud_data['country'] = 'Unknown'
        return fraud_data

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create transaction frequency, velocity, and time-based features."""
    try:
        if 'user_id' in df.columns:
            df['transaction_count_user'] = df.groupby('user_id')['user_id'].transform('count')
        if 'device_id' in df.columns:
            df['transaction_count_device'] = df.groupby('device_id')['device_id'].transform('count')
        if 'user_id' in df.columns and 'purchase_time' in df.columns:
            df = df.sort_values(['user_id', 'purchase_time'])
            df['time_diff'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 3600
            df['avg_time_between_txn'] = df.groupby('user_id')['time_diff'].transform('mean')
            df['avg_time_between_txn'].fillna(0, inplace=True)
        if 'purchase_time' in df.columns:
            df['hour_of_day'] = df['purchase_time'].dt.hour
            df['day_of_week'] = df['purchase_time'].dt.dayofweek
        if 'purchase_time' in df.columns and 'signup_time' in df.columns:
            df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        logging.info(f"Engineered features. New columns: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Error engineering features: {e}")
    return df

def transform_data(fraud_data: pd.DataFrame, creditcard: pd.DataFrame, config: Dict[str, Any]) -> Tuple:
    """Encode, split, scale, balance, and save processed data."""
    try:
        # Dynamically select categorical columns
        fraud_cat_cols = fraud_data.select_dtypes(include='object').columns.difference(['user_id', 'device_id', 'signup_time', 'purchase_time', 'ip_address', 'time_diff'])
        fraud_data = pd.get_dummies(fraud_data, columns=fraud_cat_cols, drop_first=True)
        logging.info(f"Encoded categorical features. New columns: {fraud_data.columns.tolist()}")
        # Drop columns dynamically
        drop_cols = [col for col in ['class', 'user_id', 'device_id', 'signup_time', 'purchase_time', 'ip_address', 'time_diff'] if col in fraud_data.columns]
        X_fraud = fraud_data.drop(drop_cols, axis=1)
        y_fraud = fraud_data['class'] if 'class' in fraud_data.columns else None
        X_credit = creditcard.drop(['Class'], axis=1) if 'Class' in creditcard.columns else creditcard
        y_credit = creditcard['Class'] if 'Class' in creditcard.columns else None
        # Train-test split
        test_size = config['model']['train_test_split']['test_size']
        random_state = config['model']['train_test_split']['random_state']
        X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(
            X_fraud, y_fraud, test_size=test_size, random_state=random_state, stratify=y_fraud
        )
        X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(
            X_credit, y_credit, test_size=test_size, random_state=random_state, stratify=y_credit
        )
        logging.info(f"Train-test split completed. Fraud train shape: {X_fraud_train.shape}, Credit train shape: {X_credit_train.shape}")
        # Scale numerical features
        scaler_fraud = StandardScaler()
        X_fraud_train_scaled = scaler_fraud.fit_transform(X_fraud_train)
        X_fraud_test_scaled = scaler_fraud.transform(X_fraud_test)
        scaler_credit = StandardScaler()
        X_credit_train_scaled = scaler_credit.fit_transform(X_credit_train)
        X_credit_test_scaled = scaler_credit.transform(X_credit_test)
        # Ensure model and processed data directories exist
        os.makedirs('models', exist_ok=True)
        os.makedirs(os.path.dirname(config['data']['processed']['fraud_train_smote']), exist_ok=True)
        # Save scalers for deployment
        joblib.dump(scaler_fraud, 'models/scaler_fraud.pkl')
        joblib.dump(scaler_credit, 'models/scaler_credit.pkl')
        logging.info("Saved scalers to models/")
        # Apply SMOTE
        smote = SMOTE(random_state=config['model']['smote']['random_state'])
        X_fraud_train_smote, y_fraud_train_smote = smote.fit_resample(X_fraud_train_scaled, y_fraud_train)
        X_credit_train_smote, y_credit_train_smote = smote.fit_resample(X_credit_train_scaled, y_credit_train)
        logging.info(f"Applied SMOTE. Fraud train class distribution:\n{pd.Series(y_fraud_train_smote).value_counts()}")
        logging.info(f"Credit train class distribution:\n{pd.Series(y_credit_train_smote).value_counts()}")
        # Save processed data
        pd.DataFrame(X_fraud_train_smote, columns=X_fraud.columns).to_csv(config['data']['processed']['fraud_train_smote'], index=False)
        pd.DataFrame(y_fraud_train_smote, columns=['class']).to_csv(config['data']['processed']['fraud_train_labels_smote'], index=False)
        pd.DataFrame(X_fraud_test_scaled, columns=X_fraud.columns).to_csv(config['data']['processed']['fraud_test'], index=False)
        pd.DataFrame(y_fraud_test, columns=['class']).to_csv(config['data']['processed']['fraud_test_labels'], index=False)
        pd.DataFrame(X_credit_train_smote, columns=X_credit.columns).to_csv(config['data']['processed']['credit_train_smote'], index=False)
        pd.DataFrame(y_credit_train_smote, columns=['Class']).to_csv(config['data']['processed']['credit_train_labels_smote'], index=False)
        pd.DataFrame(X_credit_test_scaled, columns=X_credit.columns).to_csv(config['data']['processed']['credit_test'], index=False)
        pd.DataFrame(y_credit_test, columns=['Class']).to_csv(config['data']['processed']['credit_test_labels'], index=False)
        logging.info("Saved processed datasets to data/processed/")
        return (X_fraud_train_smote, y_fraud_train_smote, X_fraud_test_scaled, y_fraud_test,
                X_credit_train_smote, y_credit_train_smote, X_credit_test_scaled, y_credit_test,
                scaler_fraud, scaler_credit)
    except Exception as e:
        logging.error(f"Error in transform_data: {e}")
        raise

def main(config: Dict[str, Any]):
    """Main function to run all preprocessing steps."""
    try:
        # Load datasets
        fraud_data = pd.read_csv(config['data']['raw']['fraud'])
        ip_country = pd.read_csv(config['data']['raw']['ip_country'])
        creditcard = pd.read_csv(config['data']['raw']['creditcard'])
        # Apply preprocessing steps
        fraud_data = handle_missing_values(fraud_data, 'fraud_data')
        ip_country = handle_missing_values(ip_country, 'ip_country')
        creditcard = handle_missing_values(creditcard, 'creditcard')
        fraud_data = clean_data(fraud_data, 'fraud_data')
        ip_country = clean_data(ip_country, 'ip_country')
        creditcard = clean_data(creditcard, 'creditcard')
        fraud_data = merge_ip_country(fraud_data, ip_country)
        fraud_data = engineer_features(fraud_data)
        # Save processed fraud data
        os.makedirs(os.path.dirname(config['data']['processed']['fraud']), exist_ok=True)
        fraud_data.to_csv(config['data']['processed']['fraud'], index=False)
        creditcard.to_csv(config['data']['processed']['creditcard'], index=False)
        # Transform data
        transformed_data = transform_data(fraud_data, creditcard, config)
        return transformed_data
    except Exception as e:
        logging.error(f"Error in main preprocessing pipeline: {e}")
        raise

if __name__ == "__main__":
    import yaml
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)