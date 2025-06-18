from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI
import logging
import numpy as np
from ai import generate_response
from sklearn.metrics.pairwise import cosine_distances
from embedding import get_embedding
import os
import uvicorn


# Configure logging at the start of your application
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO)
# Acquire logger
logger = logging.getLogger("fastapi")
load_dotenv()
app = FastAPI()

#loading all data and embeddings
def load_files():
    # Load embeddings and texts from the npz file
    # TODO: Make this path configurable or use a more dynamic approach
    #path ="D:\\dev\\tds-projects\\llm-teaching-assisst\\embeddings-website.npz"
    path_to_website_embeddings = "embeddings-website-minillm.npz"
    path_to_discourse_embeddings = "embeddings-discourse-minillm.npz"

    data = np.load(path_to_website_embeddings)

    # this function will load the names of the arrays in the .npz file
    embeddings = data['embeddings']
    texts = data['texts']
    urls = data['urls']

    # load the discourse data
    data = np.load(path_to_discourse_embeddings)
    embeddings = [*embeddings, *data['embeddings']]
    texts = [*texts, *data['texts']]
    urls = [*urls, *data['urls']]
    return urls, texts, embeddings

all_urls, all_texts, all_embeddings = load_files()
logger.info(f"------loaded all embeddings, texts and urls -----")
logger.info(f"Total number of urls: {len(all_urls)}")
logger.info(f"Total number of texts: {len(all_texts)}")
logger.info(f"Total number of embeddings: {len(all_embeddings)}")
#------------------end of loading data------------------


#------------------adding semantic search------------------
def semantic_search(question, urls, texts, embeddings, top_k=10):
    """
    Search for the most relevant text chunks for the question.
    """
    question_embedding = get_embedding(question)
    distances = cosine_distances([question_embedding], embeddings)[0]
    top_indices = distances.argsort()[:top_k]
    return [{"url": urls[i], "text": texts[i]} for i in top_indices]

@app.get("/")
def root():
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

@app.get("/test2")
def test_ai2():
    """
    Test the AI functionality by calling the AI model with a sample question.
    """
   
    question = "should I use podman or docker for my project?"

    logger.info("--------------------   --------------------")
    #get top 5 documents
    documents = semantic_search(question, all_urls, all_texts, all_embeddings, top_k=5)    
    logger.info(f"Retrieved {len(documents)} relevant documents for the question.")
    # neatly print the documents
    for doc in documents:
        logger.info(f"Document URL: {doc['url']}")
        logger.info(f"Document Text: {doc['text']}")

    logger.info("--------------------   --------------------")

    response = generate_response(question, documents= documents)
    logger.info(f"AI Response: {response}")
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)