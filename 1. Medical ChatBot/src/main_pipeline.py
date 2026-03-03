import os
from dotenv import load_dotenv
from src.utils.cleaning import clean_text
from src.utils.chunking import chunk_and_save_text
from src.database.vectorstore import embed_chunks, upload_to_pinecone, create_rag_chain

# Load environment variables
load_dotenv()

def run_preprocessing():
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, 'data', 'processed', 'medical_text.txt')
    cleaned_path = os.path.join(project_root, 'data', 'processed', 'medical_text_cleaned.txt')
    chunked_output_path = os.path.join(project_root, 'data', 'processed', 'medical_chunks.txt')
    embedding_path = os.path.join(project_root, 'data', 'processed', 'medical_chunks_embeddings.npy')

    # Ensure processed directory exists
    os.makedirs(os.path.join(project_root, 'data', 'processed'), exist_ok=True)

    # Clean and chunk text
    clean_text(input_path, cleaned_path)
    chunks = chunk_and_save_text(cleaned_path, chunked_output_path)

    # Embed chunks
    chunks, embeddings = embed_chunks(chunked_output_path, embedding_path)

    # Pinecone parameters from environment variables
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot-index")

    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY not found in environment variables.")
        return

    # Upload to Pinecone
    upload_to_pinecone(embeddings, chunks, INDEX_NAME, PINECONE_API_KEY)

    # RAG pipeline setup check
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if HUGGINGFACEHUB_API_TOKEN:
        qa_chain = create_rag_chain(
            INDEX_NAME,
            PINECONE_API_KEY,
            "sentence-transformers/all-MiniLM-L6-v2",
            HUGGINGFACEHUB_API_TOKEN
        )
        print("RAG chain successfully initialized.")
    else:
        print("HUGGINGFACEHUB_API_TOKEN not found, skipping RAG chain initialization check.")

if __name__ == "__main__":
    run_preprocessing()
