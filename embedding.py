import json
from sentence_transformers import SentenceTransformer
import numpy as np

def chunk_text(text, chunk_size=250):
    """
    Split text into chunks of specified size.

    Args:
        text (str): The text to split.
        chunk_size (int): Number of words per chunk.

    Returns:
        List[str]: List of text chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_embedding(chunk):
    """
    Generate embedding for a piece of text using a sentence transformer.

    Args:
        chunk (str): The text to embed.

    Returns:
        List[float]: The embedding vector for the input text.
    """
    model_name='all-MiniLM-L6-v2'

    model = SentenceTransformer(model_name)
    embedding = model.encode(chunk)

    return embedding

def process_files():

    "Process files by reading, chunking, embedding, and saving."
    # Load the JSON data
    # TODO : add relative path below
    path  = "../../data/tds_data.json"

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # chunking and embedding from course website while keeping track of urls

    embeddings_list = [] #List of dictionaries with "url","text" and embeddings
    #test_count = 0

    for item in data:
        #test_count += 1
        text = item["text"]
        url = item["url"]
        text_chunks = chunk_text(text)  
        for chunk in text_chunks:
            print(f"\t\tEmbedding chunk of size {len(chunk)}")
            embeddings_list.append({ "url": url, "text": chunk, "embedding": get_embedding(chunk)})
            print(f"\t\tEmbedding size: {len(embeddings_list[-1]['embedding'])}")
    
        """ if test_count == 5:
            print(f"Processed {test_count} items, stopping for testing.")
            break """
    
    np.savez( "embeddings-website.npz", urls=[item['url'] for item in embeddings_list],
             texts=[item['text'] for item in embeddings_list],
             embeddings=[item['embedding'] for item in embeddings_list] )
    
if __name__ == "__main__":
    process_files()
    print("Processing complete. Embeddings saved.")