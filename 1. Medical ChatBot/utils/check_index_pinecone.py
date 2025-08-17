import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Charger les variables d'environnement
load_dotenv()

# Récupérer la clé API et le nom de l'index
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_KEY = "pcsk_59j9bi_5kZifEvnXRfhXngNU22mGYPGBs8He2cTdTy3kKWzuT3w5dUByy6j3iuUiJEnCFt"
INDEX_NAME = "medical-chatbot-index"  

# Connexion à Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Vérifier si l'index existe
if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    print(f"L'index '{INDEX_NAME}' n'existe pas.")
else:
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    vector_count = stats.get("total_vector_count", 0)

    if vector_count > 0:
        print(f"L'index '{INDEX_NAME}' contient {vector_count} vecteurs.")
    else:
        print(f"L'index '{INDEX_NAME}' est vide.")
