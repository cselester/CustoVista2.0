import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import FRAUD_DETECTED_DATA_PATH, DATA_DIR

class FraudDetectionAgent:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.customer_fraud_path = FRAUD_DETECTED_DATA_PATH
        self.transaction_fraud_path = os.path.join(self.data_dir, "fraud_detected_transactions.csv")

    def detect_transaction_fraud(self, transaction_df: pd.DataFrame) -> pd.DataFrame:
        print("Running enhanced transaction fraud detection...")
        df = transaction_df.copy()
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['hour'] = df['transaction_date'].dt.hour
        df['day_of_week'] = df['transaction_date'].dt.dayofweek

        # Daypart assignment
        def assign_daypart(hour):
            if 5 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 21:
                return 'evening'
            else:
                return 'night'

        df['daypart'] = df['hour'].apply(assign_daypart)

        # Customer stats
        customer_stats = df.groupby('customer_id').agg({
            'amount': ['mean', 'std', 'count', 'max']
        }).reset_index()
        customer_stats.columns = ['customer_id', 'avg_amount', 'std_amount', 'transaction_count', 'max_amount']
        df = df.merge(customer_stats, on='customer_id', how='left')

        # Z-score
        df['amount_zscore'] = (df['amount'] - df['avg_amount']) / df['std_amount'].replace(0, 1)

        # Velocity features
        df = df.sort_values(['customer_id', 'transaction_date'])
        df['prev_transaction_time'] = df.groupby('customer_id')['transaction_date'].shift(1)
        df['time_since_last_txn'] = (df['transaction_date'] - df['prev_transaction_time']).dt.total_seconds().fillna(0)

        # One-hot encode daypart
        df = pd.get_dummies(df, columns=['daypart'], drop_first=True)

        # Features for detection
        features = ['amount', 'amount_zscore', 'hour', 'day_of_week', 'time_since_last_txn']
        features += [col for col in df.columns if col.startswith('daypart_')]
        X = df[features].fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso_forest = IsolationForest(contamination=0.03, random_state=42)
        df['anomaly_score'] = iso_forest.fit_predict(X_scaled)
        df['is_fraudulent'] = df['anomaly_score'] == -1

        # Rule-based enhancements
        df.loc[(df['amount'] > 5 * df['avg_amount']) & (df['amount'] > 3 * df['max_amount']), 'is_fraudulent'] = True

        # Unusual hours
        customer_hour_counts = df.groupby(['customer_id', 'hour']).size().reset_index(name='count')
        unusual_customers = customer_hour_counts[
            (customer_hour_counts['hour'].between(2, 4)) &
            (customer_hour_counts['count'] < 3)
        ]['customer_id'].unique()
        df.loc[(df['customer_id'].isin(unusual_customers)) &
               (df['hour'].between(2, 4)), 'is_fraudulent'] = True

        # Dynamic z-score threshold
        high_z = df['amount_zscore'] > df['amount_zscore'].mean() + 3 * df['amount_zscore'].std()
        df.loc[high_z, 'is_fraudulent'] = True

        print(f"Total transactions: {len(df)}")
        print(f"Transactions flagged as fraudulent: {df['is_fraudulent'].sum()}")
        print(f"Fraud rate: {df['is_fraudulent'].sum() / len(df):.2%}")

        return df

    def calculate_customer_fraud_risk(self, customer_df: pd.DataFrame, transaction_df: pd.DataFrame) -> pd.DataFrame:
        print("Calculating enhanced customer fraud risk...")
        df = customer_df.copy()
        transaction_df['transaction_date'] = pd.to_datetime(transaction_df['transaction_date'])

        fraud_metrics = transaction_df.groupby('customer_id').agg({
            'is_fraudulent': ['sum', 'mean'],
            'amount': ['mean', 'max'],
            'transaction_id': 'count'
        }).reset_index()
        fraud_metrics.columns = [
            'customer_id', 'fraudulent_count', 'fraud_rate',
            'avg_amount', 'max_amount', 'transaction_count'
        ]
        df = df.merge(fraud_metrics, on='customer_id', how='left').fillna(0)

        # Days since last fraud
        fraud_txns = transaction_df[transaction_df['is_fraudulent']]
        last_fraud_time = fraud_txns.groupby('customer_id')['transaction_date'].max().reset_index()
        last_fraud_time.columns = ['customer_id', 'last_fraud_time']
        now = datetime.now()
        last_fraud_time['days_since_last_fraud'] = (now - last_fraud_time['last_fraud_time']).dt.days
        df = df.merge(last_fraud_time[['customer_id', 'days_since_last_fraud']], on='customer_id', how='left')
        df['days_since_last_fraud'] = df['days_since_last_fraud'].fillna(999)

        # Raw score
        df['fraud_risk_score'] = (
            0.5 * df['fraud_rate'] +
            0.3 * (df['max_amount'] / 10000).clip(0, 1) +
            0.2 * (df['transaction_count'] / 100).clip(0, 1)
        )
        # Recent fraud adjustment
        df['fraud_risk_score'] += 0.1 * (1 - (df['days_since_last_fraud'] / 365).clip(0, 1))

        # Normalize
        min_score = df['fraud_risk_score'].min()
        max_score = df['fraud_risk_score'].max()
        df['fraud_risk_score'] = (df['fraud_risk_score'] - min_score) / (max_score - min_score + 1e-5)

        df['is_high_risk'] = (
            (df['fraud_risk_score'] > 0.4) |
            (df['fraudulent_count'] >= 3) |
            (df['fraud_rate'] > 0.3)
        )

        print(f"Total customers: {len(df)}")
        print(f"High risk customers: {df['is_high_risk'].sum()}")

        return df

    def save_results(self, customer_df: pd.DataFrame, transaction_df: pd.DataFrame):
        customer_df.to_csv(self.customer_fraud_path, index=False)
        transaction_df.to_csv(self.transaction_fraud_path, index=False)
        return {
            "customer_fraud_path": self.customer_fraud_path,
            "transaction_fraud_path": self.transaction_fraud_path
        }

    def run(self, customer_df: pd.DataFrame, transaction_df: pd.DataFrame):
        if customer_df.empty or transaction_df.empty:
            print("No data provided.")
            return {"status": "error", "message": "Empty data."}

        print(f"Starting fraud detection process...")
        print(f"Processing {len(customer_df)} customers and {len(transaction_df)} transactions")

        fraud_transaction_df = self.detect_transaction_fraud(transaction_df)
        fraudulent_count = fraud_transaction_df['is_fraudulent'].sum()

        fraud_customer_df = self.calculate_customer_fraud_risk(customer_df, fraud_transaction_df)
        high_risk_count = fraud_customer_df['is_high_risk'].sum()

        output_paths = self.save_results(fraud_customer_df, fraud_transaction_df)

        fraud_stats = {
            "total_transactions": len(transaction_df),
            "fraudulent_transactions": fraudulent_count,
            "fraud_rate": fraudulent_count / len(transaction_df),
            "high_risk_customers": high_risk_count,
            "high_risk_rate": high_risk_count / len(customer_df)
        }

        print("\nFraud Statistics:")
        for key, value in fraud_stats.items():
            print(f"{key}: {value}")

        return {
            "status": "success",
            "output_paths": output_paths,
            "fraud_statistics": fraud_stats
        }

if __name__ == "__main__":
    sample_customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name": ["John", "Jane", "Bob"],
        "risk_score": [0.2, 0.8, 0.4]
    })

    sample_transactions = pd.DataFrame({
        "transaction_id": [101, 102, 103, 104],
        "customer_id": [1, 2, 1, 3],
        "amount": [5000, 15000, 200, 4500],
        "transaction_date": ["2024-01-01 10:00:00", "2024-01-01 03:00:00", "2024-01-02 14:00:00", "2024-01-02 02:00:00"]
    })

    agent = FraudDetectionAgent()
    result = agent.run(sample_customers, sample_transactions)
    print(result)
