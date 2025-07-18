import pytest
import pandas as pd
import os
from src.data_preprocessing import handle_missing_values, clean_data, merge_ip_country, engineer_features, transform_data

def test_handle_missing_values():
    """Test missing value handling for fraud_data."""
    data = pd.DataFrame({
        'source': ['SEO', None, 'Ads'],
        'purchase_value': [100, None, 200],
        'class': [0, 1, None]
    })
    result = handle_missing_values(data.copy(), 'fraud_data')
    assert result['source'].isnull().sum() == 0
    assert result['purchase_value'].isnull().sum() == 0
    assert len(result) == 2
    assert result['source'].iloc[1] == 'SEO'  # Mode imputation
    assert result['purchase_value'].iloc[1] == 150  # Median imputation

def test_clean_data():
    """Test duplicate removal and type conversion for fraud_data."""
    data = pd.DataFrame({
        'signup_time': ['2023-01-01', '2023-01-01'],
        'purchase_time': ['2023-01-02', '2023-01-02'],
        'ip_address': ['192168001001', '192168001001']
    })
    result = clean_data(data.copy(), 'fraud_data')
    assert len(result) == 1
    assert pd.api.types.is_datetime64_any_dtype(result['signup_time'])
    assert pd.api.types.is_int64_dtype(result['ip_address'])

def test_merge_ip_country():
    """Test IP to country mapping."""
    fraud_data = pd.DataFrame({'ip_address': [192168001001]})
    ip_country = pd.DataFrame({
        'lower_bound_ip_address': [192168001000],
        'upper_bound_ip_address': [192168001999],
        'country': ['USA']
    })
    result = merge_ip_country(fraud_data.copy(), ip_country)
    assert result['country'].iloc[0] == 'USA'
    assert 'Unknown' not in result['country'].values

def test_engineer_features():
    """Test feature engineering for transaction and time-based features."""
    data = pd.DataFrame({
        'user_id': [1, 1, 2],
        'device_id': ['d1', 'd1', 'd2'],
        'purchase_time': pd.to_datetime(['2023-01-02 10:00', '2023-01-02 11:00', '2023-01-03 12:00']),
        'signup_time': pd.to_datetime(['2023-01-01 09:00', '2023-01-01 09:00', '2023-01-02 10:00'])
    })
    result = engineer_features(data.copy())
    assert result['transaction_count_user'].iloc[0] == 2
    assert result['transaction_count_device'].iloc[0] == 2
    assert result['avg_time_between_txn'].iloc[0] == 1.0
    assert result['hour_of_day'].iloc[0] == 10
    assert result['time_since_signup'].iloc[0] == 25.0

@pytest.mark.usefixtures("tmp_path")
def test_transform_data(tmp_path):
    """Test transform_data with file outputs using a temporary directory."""
    fraud_data = pd.DataFrame({
        'user_id': [1, 2], 'device_id': ['d1', 'd2'], 'signup_time': pd.to_datetime(['2023-01-01', '2023-01-01']),
        'purchase_time': pd.to_datetime(['2023-01-02', '2023-01-02']), 'ip_address': [192168001001, 192168001002],
        'source': ['SEO', 'Ads'], 'browser': ['Chrome', 'Safari'], 'sex': ['M', 'F'], 'purchase_value': [100, 200],
        'age': [30, 40], 'class': [0, 1], 'country': ['USA', 'UK']
    })
    creditcard = pd.DataFrame({
        'Time': [0, 1], 'V1': [1.0, 2.0], 'Amount': [100, 200], 'Class': [0, 1]
    })
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    config = {
        'model': {
            'train_test_split': {'test_size': 0.5, 'random_state': 42},
            'smote': {'random_state': 42}
        },
        'data': {
            'processed': {
                'fraud_train_smote': str(processed_dir / 'fraud_train_smote.csv'),
                'fraud_train_labels_smote': str(processed_dir / 'fraud_train_labels_smote.csv'),
                'fraud_test': str(processed_dir / 'fraud_test.csv'),
                'fraud_test_labels': str(processed_dir / 'fraud_test_labels.csv'),
                'credit_train_smote': str(processed_dir / 'credit_train_smote.csv'),
                'credit_train_labels_smote': str(processed_dir / 'credit_train_labels_smote.csv'),
                'credit_test': str(processed_dir / 'credit_test.csv'),
                'credit_test_labels': str(processed_dir / 'credit_test_labels.csv')
            }
        }
    }
    result = transform_data(fraud_data.copy(), creditcard.copy(), config)
    assert len(result) == 10  # 8 data arrays + 2 scalers
    assert result[0].shape[0] >= result[2].shape[0]  # SMOTE increases training size
    # Check that files are created
    for key, path in config['data']['processed'].items():
        assert os.path.exists(path)