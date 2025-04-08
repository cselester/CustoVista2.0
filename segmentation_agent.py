# segmentation_agent.py
import pandas as pd
import numpy as np
from datetime import datetime

class SegmentationAgent:
    def __init__(self):
        pass

    def segment_customers(self, customer_df: pd.DataFrame, transaction_df: pd.DataFrame) -> pd.DataFrame:
        """Segment customers using RFM logic (Recency, Frequency, Monetary)"""
        print("Running customer segmentation...")

        # Make sure transaction_date is datetime
        transaction_df['transaction_date'] = pd.to_datetime(transaction_df['transaction_date'])

        # Reference date for recency calculation
        current_date = transaction_df['transaction_date'].max()

        # RFM calculations
        rfm = transaction_df.groupby('customer_id').agg({
            'transaction_date': lambda x: (current_date - x.max()).days,  # Recency
            'transaction_id': 'count',  # Frequency
            'amount': 'sum'  # Monetary
        }).reset_index()

        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

        # Scoring
        rfm['recency_score'] = pd.qcut(rfm['recency'], q=4, labels=[4, 3, 2, 1]).astype(int)
        rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)
        rfm['monetary_score'] = pd.qcut(rfm['monetary'], q=4, labels=[1, 2, 3, 4]).astype(int)

        # Combine into a single RFM score
        rfm['rfm_score'] = rfm['recency_score'] + rfm['frequency_score'] + rfm['monetary_score']

        # Assign segments based on RFM score
        def assign_segment(score):
            if score >= 10:
                return 'High Value'
            elif score >= 7:
                return 'Loyal'
            elif score >= 5:
                return 'Promising'
            else:
                return 'At Risk'

        rfm['segment'] = rfm['rfm_score'].apply(assign_segment)

        # Merge with customer data
        segmented_df = customer_df.merge(rfm, on='customer_id', how='left')

        # Fill missing segments (e.g., customers with no transactions)
        segmented_df['segment'] = segmented_df['segment'].fillna('New')

        # Print segment stats
        print("\nCustomer Segmentation Summary:")
        print(segmented_df['segment'].value_counts())

        return segmented_df
