import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec

def embed_chunks(chunked_output_path, embedding_path, model_name='all-MiniLM-L6-v2'):
    """
    Embeds text chunks using a SentenceTransformer model.
    """
    try:
        with open(chunked_output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        chunks = re.split(r'--- Chunk \d+ ---\n', content)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        embedding_model = SentenceTransformer(model_name)
        embeddings = embedding_model.encode(chunks, show_progress_bar=True)

        np.save(embedding_path, embeddings)
        print(f"Text chunks embedded and saved to {embedding_path}")
        return chunks, embeddings
    except Exception as e:
        print(f"Error embedding chunks: {e}")
        return [], None

def upload_to_pinecone(embeddings, chunks, index_name, pinecone_api_key, batch_size=100):
    """
    Uploads embeddings and metadata to Pinecone.
    """
    try:
        pc = Pinecone(api_key=pinecone_api_key)

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="gcp", region="us-central1")
            )

        index = pc.Index(index_name)

        vectors = []
        for i, (embedding, text) in enumerate(zip(embeddings, chunks)):
            vectors.append({
                "id": f"chunk-{i + 1}",
                "values": embedding.tolist(),
                "metadata": {"text": text}
            })

        for i in range(0, len(vectors), batch_size):
            index.upsert(vectors=vectors[i:i + batch_size])
        print(f"{len(vectors)} embeddings uploaded to Pinecone index '{index_name}'.")
    except Exception as e:
        print(f"Error uploading to Pinecone: {e}")

def create_rag_chain(index_name, pinecone_api_key, embeddings_model_name, huggingfacehub_api_token):
    """
    Creates a RetrievalQA chain using Pinecone and HuggingFace.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        vectorstore = LangchainPinecone(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=pinecone_api_key
        )

        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.1, "max_length": 512},
            huggingfacehub_api_token=huggingfacehub_api_token,
            task="text2text-generation"
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        return qa_chain
    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        return None
