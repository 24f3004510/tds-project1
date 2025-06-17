from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI
import logging

from ai import generate_response


# Configure logging at the start of your application
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO)
# Acquire logger
logger = logging.getLogger("fastapi")
load_dotenv()
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello Monica"}

@app.get("/test")
def test_ai():
    """
    Test the AI functionality by calling the AI model with a sample question.
    """
    documents = [{
        "url": "https://example.com/doc-1",
        "text": "My favorite color is blue. It is a calming and peaceful color.",
    },
    {
        "url": "https://example.com/doc-2",
        "text": "jay's favorite color is red. It is a bold and vibrant color."
    }]
    question = "what is Usha's favorite color?"
    response = generate_response(question, documents= documents)
    logger.info(f"AI Response: {response}")
    return response