# oversight_panel_agent.py
import pandas as pd

class OversightPanelAgent:
    def __init__(self, escalation_threshold=0.6):
        self.escalation_threshold = escalation_threshold

    def review_high_risk_cases(self, customer_df: pd.DataFrame, transaction_df: pd.DataFrame) -> pd.DataFrame:
        print("Running oversight panel agent for case review and escalation...")

        # Filter high-risk customers
        high_risk_customers = customer_df[customer_df['is_high_risk'] == True].copy()

        # Merge to bring transaction context
        merged_df = transaction_df.merge(
            high_risk_customers[['customer_id', 'fraud_risk_score']],
            on='customer_id', how='inner'
        )

        # Flag transactions with high fraud score or high amount
        merged_df['escalate_case'] = (
            (merged_df['fraud_risk_score'] > self.escalation_threshold) |
            (merged_df['amount'] > 10000) |
            (merged_df['is_fraudulent'] == True)
        )

        # Extract escalated cases
        escalated_cases = merged_df[merged_df['escalate_case'] == True]

        print(f"Total high-risk customers: {len(high_risk_customers)}")
        print(f"Cases escalated for human review: {len(escalated_cases)}")

        return escalated_cases[['customer_id', 'transaction_id', 'amount', 'transaction_date', 'fraud_risk_score', 'is_fraudulent']]

if __name__ == "__main__":
    # Test with dummy data if needed
    print("Oversight Panel Agent loaded successfully.")
