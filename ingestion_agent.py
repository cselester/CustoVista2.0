import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

class IngestionAgent:
    def __init__(self, num_customers=100, num_transactions=1000):
        self.fake = Faker()
        self.num_customers = num_customers
        self.num_transactions = num_transactions

    def generate_customers(self) -> pd.DataFrame:
        """Generate synthetic customer data"""
        customers = []
        for i in range(1, self.num_customers + 1):
            customers.append({
                "customer_id": i,
                "name": self.fake.name(),
                "email": self.fake.email(),
                "phone": self.fake.phone_number(),
                "address": self.fake.address().replace("\n", ", "),
                "account_created": self.fake.date_between(start_date='-5y', end_date='today'),
                "risk_score": round(random.uniform(0, 1), 2)
            })
        return pd.DataFrame(customers)

    def generate_transactions(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic transaction data"""
        transactions = []
        for i in range(1, self.num_transactions + 1):
            customer = customers_df.sample(1).iloc[0]
            amount = round(random.uniform(10, 20000), 2)
            timestamp = datetime.now() - timedelta(days=random.randint(0, 365), hours=random.randint(0, 23), minutes=random.randint(0, 59))

            transactions.append({
                "transaction_id": i,
                "customer_id": customer['customer_id'],
                "amount": amount,
                "transaction_date": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "location": self.fake.city(),
                "merchant": self.fake.company()
            })
        return pd.DataFrame(transactions)

    def ingest(self):
        print("Generating synthetic customer and transaction data using Faker...")
        customer_df = self.generate_customers()
        transaction_df = self.generate_transactions(customer_df)
        print(f"Generated {len(customer_df)} customers and {len(transaction_df)} transactions")
        return customer_df, transaction_df
