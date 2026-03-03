import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA

# Global variable to store the RAG chain
_qa_chain = None

def create_rag_chain(index_name, pinecone_api_key, embeddings_model_name, huggingfacehub_api_token):
    """Creates a RetrievalQA chain using Pinecone and HuggingFace."""
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

def get_response(question):
    """Initializes (if necessary) and returns a response from the RAG chain."""
    global _qa_chain

    if _qa_chain is None:
        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        INDEX_NAME = "medical-chatbot-index"
        EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

        if not PINECONE_API_KEY or not HUGGINGFACEHUB_API_TOKEN:
            return "Error: API keys not set."

        _qa_chain = create_rag_chain(
            INDEX_NAME,
            PINECONE_API_KEY,
            EMBEDDINGS_MODEL,
            HUGGINGFACEHUB_API_TOKEN
        )

    try:
        response = _qa_chain.invoke(question)
        return response["result"]
    except Exception as e:
        return f"Error: {str(e)}"
