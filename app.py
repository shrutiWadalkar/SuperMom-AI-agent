import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns
import sklearn
import scipy

# Check and print versions of installed packages
print(f"numpy version: {np.__version__}")  # Should print 1.23.5
print(f"matplotlib version: {matplotlib.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"seaborn version: {sns.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"scipy version: {scipy.__version__}")

# Your AI agent logic below
from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
import requests
import re
from urllib.parse import quote as url_quote


# Load environment variables from .env file
load_dotenv()

# Flask app initialization
app = Flask(__name__)

# Retrieve API key securely from environment variable
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("❌ API Key not found! Check your environment variables.")

# Hugging Face Model Details
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

# Clean the response from the model to remove unwanted tokens
def clean_response(response):
    """Remove system tokens like [INST], [/INST], and <s> from the chatbot's response"""
    response = re.sub(r'<s>\s*|\[INST\]|\[/INST\]>', '', response)  # Remove <s>, [INST], [/INST]
    response = response.lstrip("> ")  # Remove any leading '>' or spaces
    return response.strip()  # Remove extra spaces or newlines


# Function to call Hugging Face API
def ask_supermom(question):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "inputs": f"<s>[INST] User: {question} [/INST]>",
        "parameters": {"max_new_tokens": 200, "temperature": 0.7, "top_p": 0.9},
    }
    try:
        response = requests.post(f"https://api-inference.huggingface.co/models/{MODEL_NAME}", headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return clean_response(result[0]["generated_text"]) if isinstance(result, list) else "Unexpected response format."
        else:
            return f"❌ API Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"❌ Network Error: {str(e)}"

# Serve HTML Page
@app.route("/")
def home():
    return render_template("index.html")

# API Endpoint
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    answer = ask_supermom(question)
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)



