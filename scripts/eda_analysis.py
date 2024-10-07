import matplotlib.pyplot as plt
import seaborn as sns


# data datasummary
def data_summary(data):
    """Provides summary information of the dataset."""
    summary = {
        'shape': data.shape,
        #'columns': data.columns.tolist(),
        #'missing_values': data.isnull().sum(),
        #'dtypes': data.dtypes
    }
    return summary

### Numerical Feature Distribution
def plot_numerical_distribution(data):
    """Plots the distribution of numerical features: 'Amount', 'Value', 'TransactionStartTime'."""
    numerical_columns = ['Amount', 'Value', 'TransactionStartTime']
    
    for column in numerical_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()

# Categorical Feature Distribution 
def plot_categorical_distribution(data):
    """Plots the distribution of categorical features."""
    categorical_columns = [
        'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
        'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 
        'ChannelId', 'PricingStrategy', 'FraudResult'
    ]
    
    for column in categorical_columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=column, data=data)
        plt.title(f'Distribution of {column}')
        plt.xticks(rotation=90)
        plt.show()

# Correlation Analysis
def plot_correlation_matrix(data, numerical_columns):
    """Plots a correlation matrix of numerical features."""
    corr_matrix = data[numerical_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

# Missing Value Analysis
def missing_value_summary(data):
    """Provides a summary of missing values."""
    missing_values = data.isnull().sum()
    return missing_values[missing_values > 0]   

# Outlier Detection
def detect_outliers(data, numerical_columns):
    """Uses box plots to detect outliers in numerical features."""
    for column in numerical_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=data[column])
        plt.title(f'Outlier detection for {column}')
        plt.show()