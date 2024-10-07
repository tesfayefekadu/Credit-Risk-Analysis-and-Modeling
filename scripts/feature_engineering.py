import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from xverse.transformer import WOE




def create_aggregate_features(data, customer_id_col, amount_col):
    """Creates aggregate features like total transaction amount, average, and transaction count per customer."""
    
    aggregate_df = data.groupby(customer_id_col).agg(
        Total_Transaction_Amount = (amount_col, 'sum'),
        Average_Transaction_Amount = (amount_col, 'mean'),
        Transaction_Count = (amount_col, 'count'),
        Std_Transaction_Amount = (amount_col, 'std')
    ).reset_index()

    return aggregate_df

def extract_temporal_features(data):
    """Extracts features like transaction hour, day, month, and year from TransactionStartTime."""
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'], format='%Y-%m-%dT%H:%M:%SZ')
    
    data['Transaction_Hour'] = data['TransactionStartTime'].dt.hour
    data['Transaction_Day'] = data['TransactionStartTime'].dt.day
    data['Transaction_Month'] = data['TransactionStartTime'].dt.month
    data['Transaction_Year'] = data['TransactionStartTime'].dt.year

    return data

def one_hot_encode(data):
    """Applies one-hot encoding to categorical columns."""
    categorical_columns = ['CurrencyCode', 'CountryCode', 'ProductCategory', 'ChannelId', 'ProviderId']
    return pd.get_dummies(data, columns=categorical_columns)


def label_encode(data, categorical_columns):
    """Applies label encoding to categorical columns."""
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    return data, label_encoders

# missing value
def handle_missing_values(data):
    """Handles missing values by imputing mean for numeric columns."""
    data['Amount'] = data['Amount'].fillna(data['Amount'].mean())
    data['Value'] = data['Value'].fillna(data['Value'].mean())
    return data

    
# Normalization
def normalize_features(data):
    """Normalizes the Amount and Value columns to a range of [0, 1]."""
    scaler = MinMaxScaler()
    data[['Amount', 'Value']] = scaler.fit_transform(data[['Amount', 'Value']])
    return data


# Standardization
def standardize_features(data):
    """Standardizes the Amount and Value columns to have mean 0 and standard deviation 1."""
    scaler = StandardScaler()
    data[['Amount', 'Value']] = scaler.fit_transform(data[['Amount', 'Value']])
    return data



