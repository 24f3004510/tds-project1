import glob
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
import openai
import logging

# Configure logging at the start of your application
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO)

# Acquire logger
logger = logging.getLogger("ai")

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

def generate_response(question, image=None, documents=None):
    """Generate a parsed response (JSON) with answer and links using Gemini."""
    context = "Context documents:\n\n" + \
              "\n\n---\n\n".join([f"URL: {item['url']}\n{item['text']}" for item in documents])

    prompt = f"""{context}

Question: {question}

{'Additionally, here is an image you may use for context:\n' + image if image else ''}

Please respond in **JSON format** with the following structure:

{{
  "answer": "<your answer here as a string>",
  "links": [
    "<URL 1>",
    "<URL 2>"
  ]
}}

Your response should be **strictly in this format** with a valid JSON object. Do not include any additional text outside the object. Only include URLs that are relevant to the answer, and ensure the answer is concise and directly addresses the question asked. Do NOT use any information that is NOT in the context provided.
"""    
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")  # or gemini-1.5
    response = model.generate_content(prompt)
    raw_response = response.text.strip()
    logger.info("-----Raw response:", raw_response)
    try:
        raw_response = raw_response.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw_response)
        return parsed
    except json.JSONDecodeError as e:
        print("JSON parsing error!", e)
        return {"answer": "Error parsing response.", "links": []}
