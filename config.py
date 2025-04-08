import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

CUSTOMER_DATA_PATH = os.path.join(DATA_DIR, 'customers.csv')
TRANSACTION_DATA_PATH = os.path.join(DATA_DIR, 'transactions.csv')
SEGMENTED_DATA_PATH = os.path.join(DATA_DIR, 'segmented_customers.csv')
FRAUD_DETECTED_DATA_PATH = os.path.join(DATA_DIR, 'fraudulent_transactions.csv')
RECOMMENDATION_DATA_PATH = os.path.join(DATA_DIR, 'recommendations.csv')
ESCALATED_CASES_PATH = os.path.join(DATA_DIR, 'escalated_cases.csv')
