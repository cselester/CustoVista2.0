# recommendation_agent.py
import pandas as pd

class RecommendationAgent:
    def __init__(self):
        # Define basic product categories
        self.products = {
            "premium_card": "Platinum Credit Card",
            "basic_card": "Standard Debit Card",
            "loan_offer": "Personal Loan Offer",
            "investment_plan": "Wealth Investment Plan",
            "retention_offer": "Loyalty Bonus / Cashback",
            "starter_bundle": "Welcome Package for New Customers"
        }

    def generate_recommendations(self, customer_df: pd.DataFrame) -> pd.DataFrame:
        print("Running personalized product recommendation engine...")

        df = customer_df.copy()

        # Initialize recommendation column
        df["recommended_product"] = "None"

        # Assign recommendations based on segment and risk
        for i, row in df.iterrows():
            segment = row.get("segment", "Unknown")
            high_risk = row.get("is_high_risk", False)

            if high_risk:
                df.at[i, "recommended_product"] = self.products["retention_offer"]
            elif segment == "High Value":
                df.at[i, "recommended_product"] = self.products["premium_card"]
            elif segment == "Loyal":
                df.at[i, "recommended_product"] = self.products["investment_plan"]
            elif segment == "Promising":
                df.at[i, "recommended_product"] = self.products["loan_offer"]
            elif segment == "At Risk":
                df.at[i, "recommended_product"] = self.products["retention_offer"]
            elif segment == "New":
                df.at[i, "recommended_product"] = self.products["starter_bundle"]
            else:
                df.at[i, "recommended_product"] = self.products["basic_card"]

        print("\nProduct Recommendation Summary:")
        print(df["recommended_product"].value_counts())

        return df
