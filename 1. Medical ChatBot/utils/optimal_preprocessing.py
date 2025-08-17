import os
import re
import numpy as np
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from langchain.vertorstore import VectorStore
from langchain.vectorstores import PineconeVectorStore

# Charger les variables d'environnement depuis un fichier .env
load_dotenv()

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------
# Step 1: Clean the text
def clean_text(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        text = file.read()
    text = re.sub(r'\n--- Page \d+ ---\n', '\n', text)
    text = re.sub(r"GALE ENCLYCLOPEDIA OF MEDICINE 2\s*\w*\n*", '', text, flags=re.IGNORECASE)
    text = re.sub(r"Copyright © \d{4}.*\n", "", text)
    text = re.sub(r"ISBN [\d\-\(\) ]+\n*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n?\s*\d+\s*\n", "\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Text cleaned and saved to {output_path}")

# ---------------------------
# Step 2: Split text into chunks
def chunk_and_save_text(input_path, output_path, chunk_size=1000, chunk_overlap=200):
    with open(input_path, 'r', encoding='utf-8') as file:
        text = file.read()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks = splitter.split_text(text)
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i + 1} ---\n{chunk}\n\n")
    print(f"Text split into {len(chunks)} chunks and saved to {output_path}")

# ---------------------------
# Step 3: Embedding the text chunks
def embed_chunks(chunked_output_path, embedding_path):
    with open(chunked_output_path, 'r', encoding='utf-8') as f:
        content = f.read()
    chunks = re.split(r'--- Chunk \d+ ---\n', content)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    np.save(embedding_path, embeddings)
    print(f"Text chunks embedded and saved to {embedding_path}")
    return chunks, embeddings

# ---------------------------
# Step 4: Save the embeddings into Pinecone
def upload_to_pinecone(embeddings, chunks, index, batch_size=100):
    vectors = [
        {
            "id": f"chunk-{i + 1}",
            "values": embedding.tolist(),
            "metadata": {"text": text}
        }
        for i, (embedding, text) in enumerate(zip(embeddings, chunks))
    ]
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])
    print(f"{len(vectors)} Embeddings uploaded to Pinecone index.")

# ---------------------------
# Step 5: Create a Pipeline RAG (RetrievalQA chain)
def create_rag_chain(index_name, pinecone_api_key, embeddings_model_name, huggingfacehub_api_token):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vectorstore = LangchainPinecone(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        model_kwargs={"max_length": 512},
        temperature=0.1,
        huggingfacehub_api_token=huggingfacehub_api_token,
        task="text2text-generation"
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

# ---------------------------
if __name__ == "__main__":
    # Paths
    input_path = os.path.join(HOME, 'data', 'processed', 'medical_text.txt')
    cleaned_path = os.path.join(HOME, 'data', 'processed', 'medical_text_cleaned.txt')
    chunked_output_path = os.path.join(HOME, 'data', 'processed', 'medical_chunks.txt')
    embedding_path = os.path.join(HOME, 'data', 'processed', 'medical_chunks_embeddings.npy')

    # Clean and chunk text
    clean_text(input_path, cleaned_path)
    chunk_and_save_text(cleaned_path, chunked_output_path)

    # Embed chunks
    chunks, embeddings = embed_chunks(chunked_output_path, embedding_path)

    # Pinecone setup
    # PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_KEY = "pcsk_59j9bi_5kZifEvnXRfhXngNU22mGYPGBs8He2cTdTy3kKWzuT3w5dUByy6j3iuUiJEnCFt"
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    INDEX_NAME = "medical-chatbot-index"

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    # Upload to Pinecone
    upload_to_pinecone(embeddings, chunks, index)

    # RAG pipeline
    qa_chain = create_rag_chain(
        INDEX_NAME,
        PINECONE_API_KEY,
        "sentence-transformers/all-MiniLM-L6-v2",
        HUGGINGFACEHUB_API_TOKEN
    )

    # Example usage
    question = "Qu'est-ce que l'hypertension artérielle ?"
    # result = qa_chain.run(question)
    result = qa_chain.invoke(question)
    print("Réponse :", result)
