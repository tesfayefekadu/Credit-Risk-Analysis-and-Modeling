# Credit-Risk-Analysis-and-Modeling

## Overview
This project develops a credit scoring model for Bati Bank's buy-now-pay-later service in partnership with an eCommerce company. The model aims to assess customer creditworthiness and predict the likelihood of default.

## Business Context
Bati Bank, a leading financial service provider, is partnering with an eCommerce company to offer a buy-now-pay-later service. This project creates a Credit Scoring Model using data provided by the eCommerce platform to evaluate potential borrowers.


## Project Structure

```plaintext

credit-risk-analysis-and-modeling/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml   # GitHub Actions
├── .gitignore              # files and folders to be ignored by git
├── requirements.txt        # contains dependencies for the project
├── README.md               # Documentation for the projects
├── src/
│   └── __init__.py
├── notebooks/
│   ├── __init__.py
|   ├──eda_analysis.ipynb               # Jupyter notebook for customer transaction data analysis 
|   ├──feature_engineering.ipynb        # Jupyter notebook for feature engineering and woa analysis 
|   ├──weight_of_evidence.ipynb       # Jupyter notebook for ml model training 
├── tests/
│   └── __init__.py
└── scripts/
    ├── __init__.py
    ├── eda_analysis.py             # script for for customer transaction data analysis 
    ├── feature_engineering.py      # Script for for feature engineering and woa analysis behavior
    ├── woe.py            # script for ml model training model
   
    
```


## Setup

1. Clone the repository:
   ```
   git clone https://github.com/tesfayefekadu/Credit-Risk-Analysis-and-Modeling.git
   cd credit-risk-scoring-model
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```


## Model Development Process
1. Exploratory Data Analysis (EDA)
2. Feature Engineering
3. Default Estimator Creation

