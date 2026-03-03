import os
from pinecone import Pinecone, ServerlessSpec
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_extraction import extract_text_from_pdf
from utils.preprocessing import clean_text, chunk_text, embed_chunks
import numpy as np

def upload_to_pinecone(embeddings, chunks, index, batch_size=100):
    """Uploads embeddings and their corresponding text chunks to Pinecone."""
    vectors = []
    for i, (embedding, text) in enumerate(zip(embeddings, chunks)):
        vectors.append({
            "id": f"chunk-{i + 1}",
            "values": embedding.tolist(),
            "metadata": {"text": text}
        })

    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])
    print(f"{len(vectors)} embeddings uploaded to Pinecone index.")

def main():
    # Configuration
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    INDEX_NAME = "medical-chatbot-index"

    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY environment variable not set.")
        return

    # Paths
    HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(HOME, 'Data', 'Medical_book.pdf')

    # Ingestion Pipeline
    print("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)

    print("Cleaning text...")
    cleaned_text = clean_text(raw_text)

    print("Chunking text...")
    chunks = chunk_text(cleaned_text)

    print("Embedding chunks...")
    embeddings = embed_chunks(chunks)

    # Pinecone setup
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        print(f"Creating index {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="gcp", region="us-central1")
        )

    index = pc.Index(INDEX_NAME)

    print("Uploading to Pinecone...")
    upload_to_pinecone(embeddings, chunks, index)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
