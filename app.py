import gradio as gr
import pandas as pd
from ingestion_agent import IngestionAgent
from segmentation_agent import SegmentationAgent
from fraud_detection_agent import FraudDetectionAgent
from recommendation_agent import RecommendationAgent
from oversight_panel_agent import OversightPanelAgent
from groq import Groq
from dotenv import load_dotenv
import os

# ========== Load .env Variables ==========
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Please set the GROQ_API_KEY environment variable in your Hugging Face Space settings")

# ========== Agents ==========
ingestion = IngestionAgent(num_customers=200, num_transactions=1000)
segmentation = SegmentationAgent()
fraud_detection = FraudDetectionAgent()
recommendation = RecommendationAgent()
oversight = OversightPanelAgent()

# ========== Global States ==========
final_df = pd.DataFrame()
escalated_df = pd.DataFrame()

# ========== Helper Functions ==========
def build_contextual_summary():
    summary = ""

    if not final_df.empty:
        fraud_count = final_df["fraud_alert"].sum()
        total = len(final_df)
        top_segments = final_df["segment"].value_counts().head(3).to_dict()
        summary += f"There are {fraud_count} high-risk customers out of {total}.\n"
        summary += f"Top customer segments are: {top_segments}.\n"

    if not escalated_df.empty:
        top_escalated = escalated_df["customer_id"].tolist()[:3]
        summary += f"Some recently escalated customer IDs are: {top_escalated}\n"

    return summary.strip()

def get_customer_details(customer_id):
    if final_df.empty:
        return "Customer data is not loaded."

    match = final_df[final_df["id"] == int(customer_id)]
    if match.empty:
        return f"No data found for customer ID {customer_id}."
    row = match.iloc[0]
    return (
        f"Customer ID: {row['id']}\n"
        f"Name: {row['name']}\n"
        f"Segment: {row['segment']}\n"
        f"Fraud Alert: {'Yes' if row['fraud_alert'] else 'No'}\n"
        f"Recommendations: {row['recommendations']}"
    )

# ========== Main Pipeline ==========
def run_pipeline():
    global final_df, escalated_df

    customers_df, transactions_df = ingestion.ingest()
    segmented = segmentation.segment_customers(customers_df, transactions_df)

    fraud_results = fraud_detection.run(segmented, transactions_df)
    fraud_checked_customers = pd.read_csv(fraud_results["output_paths"]["customer_fraud_path"])
    fraud_checked_transactions = pd.read_csv(fraud_results["output_paths"]["transaction_fraud_path"])

    recommended = recommendation.generate_recommendations(fraud_checked_customers)
    escalated_df = oversight.review_high_risk_cases(fraud_checked_customers, fraud_checked_transactions)

    final_df = recommended.rename(columns={
        "customer_id": "id",
        "is_high_risk": "fraud_alert",
        "recommended_product": "recommendations"
    })[["id", "name", "segment", "fraud_alert", "recommendations"]]

    final_df.to_csv("output/final_customer360_result.csv", index=False)
    return final_df, "output/final_customer360_result.csv"

# ========== Groq Chat Assistant ==========
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Error initializing Groq client: {str(e)}")
    groq_client = None

def respond_to_user(message, history):
    if groq_client is None:
        return history + [(message, "Groq API not initialized. Check your key.")]
    
    try:
        system_prompt = (
            "You are an intelligent assistant for analyzing customer fraud, segmentation, and recommendations. "
            "Use the data below as internal company insights to respond:\n\n"
            + build_contextual_summary()
        )

        messages = [{"role": "system", "content": system_prompt}]
        for user, assistant in history:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": message})

        if message.lower().startswith("customer "):
            customer_id = message.split(" ")[-1]
            reply = get_customer_details(customer_id)
        else:
            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=messages
            )
            reply = response.choices[0].message.content

        history.append((message, reply))
        return history

    except Exception as e:
        error_message = f"Error communicating with Groq API: {str(e)}"
        history.append((message, error_message))
        return history

# ========== Gradio UI ==========
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Customer360 AI Dashboard")
    gr.Markdown("Simulate data ingestion, segmentation, fraud detection, recommendations, and oversight in one go.")

    with gr.Row():
        run_btn = gr.Button("🚀 Run Pipeline")
        download_btn = gr.File(label="📥 Download Result CSV")

    with gr.Tabs():
        with gr.Tab("📊 Overview"):
            overview_table = gr.Dataframe()
        
        with gr.Tab("🚨 Escalated Fraud Cases"):
            fraud_table = gr.Dataframe()

        with gr.Tab("🎯 Recommendations"):
            reco_table = gr.Dataframe()

        with gr.Tab("🤖 Fraud Chat Assistant"):
            chatbot = gr.Chatbot(label="Groq Fraud Assistant", height=400)
            msg = gr.Textbox(placeholder="Ask anything about fraud risks, suspicious transactions, etc...", show_label=False)

            msg.submit(respond_to_user, inputs=[msg, chatbot], outputs=[chatbot])

    run_btn.click(fn=run_pipeline, outputs=[overview_table, download_btn])
    run_btn.click(fn=lambda: escalated_df, outputs=fraud_table)
    run_btn.click(fn=lambda: final_df[["id", "name", "segment", "recommendations"]], outputs=reco_table)

# Launch the app
if __name__ == "__main__":
    demo.launch()
