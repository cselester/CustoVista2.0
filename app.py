# gradio_ui.py

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

def filter_output(segment, fraud_alert):
    if final_df.empty:
        return pd.DataFrame()

    df = final_df.copy()
    if segment != "All":
        df = df[df["segment"] == segment]
    if fraud_alert != "All":
        df = df[df["fraud_alert"] == (fraud_alert == "Yes")]
    return df

def get_recommendations():
    if final_df.empty:
        return pd.DataFrame()
    return final_df[["id", "name", "segment", "recommendations"]]

def get_fraud_cases():
    global escalated_df
    return escalated_df if not escalated_df.empty else pd.DataFrame()

# ========== Groq Chat Assistant ==========
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Error initializing Groq client: {str(e)}")
    groq_client = None

def respond_to_user(message, history):
    if groq_client is None:
        return history + [(message, "Error: Groq API client not properly initialized. Please check your API key.")]
    
    try:
        # Convert Gradio history (list of tuples) to OpenAI-style messages
        messages = [{"role": "system", "content": "You are a helpful fraud detection assistant."}]
        for user, assistant in history:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})

        # Add latest user message
        messages.append({"role": "user", "content": message})

        # Get response from Groq
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
            with gr.Row():
                segment_filter = gr.Dropdown(choices=["All", "Loyal", "High Value", "Promising", "At Risk", "New"],
                                              label="Segment Filter", value="All")
                fraud_filter = gr.Dropdown(choices=["All", "Yes", "No"], label="Fraud Alert Filter", value="All")
            overview_table = gr.Dataframe()

        with gr.Tab("🚨 Escalated Fraud Cases"):
            fraud_table = gr.Dataframe()

        with gr.Tab("🎯 Recommendations"):
            reco_table = gr.Dataframe()

        with gr.Tab("🤖 Fraud Chat Assistant"):
            chatbot = gr.Chatbot(label="Groq Fraud Assistant", height=400)
            msg = gr.Textbox(placeholder="Ask anything about fraud risks, suspicious transactions, etc...", show_label=False)

            msg.submit(respond_to_user, inputs=[msg, chatbot], outputs=[chatbot])

    # Button actions
    run_btn.click(fn=run_pipeline, outputs=[overview_table, download_btn])
    run_btn.click(fn=get_fraud_cases, outputs=fraud_table)
    run_btn.click(fn=get_recommendations, outputs=reco_table)
    segment_filter.change(fn=filter_output, inputs=[segment_filter, fraud_filter], outputs=overview_table)
    fraud_filter.change(fn=filter_output, inputs=[segment_filter, fraud_filter], outputs=overview_table)

# Launch the app
if __name__ == "__main__":
    demo.launch()
