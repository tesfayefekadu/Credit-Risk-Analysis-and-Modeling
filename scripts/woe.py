from xverse.transformer import WOE
import pandas as pd


# Calculate RFMS Score
def calculate_rfms_score(data, recency_weight=0.25, frequency_weight=0.25, monetary_weight=0.25, severity_weight=0.25):
    # Convert TransactionStartTime to datetime if it is not already in datetime format
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'], errors='coerce')
    
    # Calculate Recency: Time since the last transaction for each customer
    data['Recency'] = (pd.Timestamp.now(tz='UTC') - data.groupby('CustomerId')['TransactionStartTime'].transform('max')).dt.days
    
    # Calculate Frequency: Total number of transactions per customer
    data['Frequency'] = data.groupby('CustomerId')['TransactionId'].transform('count')
    
    # Calculate Monetary: Total transaction amount per customer
    data['Monetary'] = data.groupby('CustomerId')['Amount'].transform('sum')
    
    # Use FraudResult as a proxy for Severity (replace with relevant metric)
    data['Severity'] = data['FraudResult']
    
    # Calculate RFMS score
    data['RFMS_Score'] = (recency_weight * data['Recency'] +
                          frequency_weight * data['Frequency'] +
                          monetary_weight * data['Monetary'] +
                          severity_weight * data['Severity'])
    
    return data
# Assign Good/Bad Labels
def assign_risk_labels(data, threshold=0.5):
    """Assigns 'good' or 'bad' label based on RFMS score."""
    
    # Normalize RFMS Score to a scale of 0 to 1
    data['RFMS_Score_Normalized'] = (data['RFMS_Score'] - data['RFMS_Score'].min()) / (data['RFMS_Score'].max() - data['RFMS_Score'].min())
    
    # Assign 'good' or 'bad' labels based on a threshold
    data['Risk_Label'] = data['RFMS_Score_Normalized'].apply(lambda x: 'good' if x >= threshold else 'bad')
    
    return data


def perform_woe_binning(data):
    """Performs WoE binning on categorical features."""
    woe_encoder = WOE()
    woe_features = ['Recency', 'Frequency', 'Monetary', 'Severity']

    # Inspect unique values in each feature to catch any non-numeric values
    for feature in woe_features:
        print(f"Unique values in {feature}: {data[feature].unique()}")

    # Convert all relevant features to numeric, forcing errors to NaN
    for feature in woe_features:
        data[feature] = pd.to_numeric(data[feature], errors='coerce')

    # Check for missing values after conversion and handle them (e.g., fill NaN or drop rows)
    print("Missing values after conversion:")
    print(data[woe_features].isnull().sum())

    # You can choose to either fill NaN values or drop rows with NaNs
    # data = data.dropna(subset=woe_features)  # Option to drop rows with NaNs
    data = data.fillna(0)  # Option to fill NaNs with 0, or other strategies like mean or median

    # Apply WoE binning based on Risk_Label
    woe_encoder.fit(data[woe_features], data['Risk_Label'])

    # Transform data using WoE encoder
    data_woe = woe_encoder.transform(data[woe_features])

    return data_woe, woe_encoder.woe_df_







