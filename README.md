---
title: Custovista
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.19.2"
app_file: app.py
pinned: false
---

# Custovista

A comprehensive customer analytics dashboard that combines data ingestion, segmentation, fraud detection, and recommendations in one unified interface.

## Features

- 🚀 One-click pipeline execution
- 📊 Customer segmentation analysis
- 🚨 Fraud detection and alerts
- 🎯 Personalized recommendations
- 🤖 AI-powered fraud chat assistant

## Usage

1. Click the "Run Pipeline" button to start the analysis
2. Use the filters to explore different customer segments
3. Check the "Escalated Fraud Cases" tab for high-risk customers
4. View personalized recommendations in the "Recommendations" tab
5. Interact with the AI assistant for fraud-related queries

## Environment Variables

The application requires a GROQ API key. Set it in your Hugging Face Space's environment variables:

- `GROQ_API_KEY`: Your Groq API key

## Technical Details

Built with:
- Python
- Gradio
- Pandas
- Scikit-learn
- Groq API 