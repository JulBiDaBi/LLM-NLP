# Medical ChatBot

Projet de ChatBot médical utilisant LangChain, Pinecone et HuggingFace.

## Structure du Projet

```text
1. Medical ChatBot/
├── data/
│   ├── raw/             # Contient le PDF original (Medical_book.pdf)
│   └── processed/       # Contient le texte extrait, les chunks et les embeddings
├── src/
│   ├── api/             # API FastAPI
│   ├── chatbot/         # Logique du chatbot (prompts, etc.)
│   ├── database/        # Gestion de la base de données vectorielle (Pinecone)
│   ├── ui/              # Interface utilisateur Streamlit
│   ├── utils/           # Fonctions utilitaires (extraction, nettoyage, chunking)
│   └── main_pipeline.py # Script pour lancer le pipeline complet
├── .env.example         # Exemple de variables d'environnement
├── .gitignore           # Fichiers à ignorer par Git
├── requirements.txt     # Dépendances du projet
├── tests/               # Tests unitaires et d'intégration
└── README.md            # Ce fichier
```

## Installation

1. Clonez le dépôt.
2. Créez un environnement virtuel : `python -m venv venv`.
3. Activez l'environnement virtuel :
   - Windows : `venv\Scripts\activate`
   - Linux/Mac : `source venv/bin/activate`
4. Installez les dépendances : `pip install -r requirements.txt`.
5. Copiez `.env.example` vers `.env` et remplissez vos clés API.

## Utilisation

### 1. Préparation des données
Lancez l'extraction de texte (si nécessaire) et le pipeline de prétraitement :
```bash
python -m src.utils.text_extraction
python -m src.main_pipeline
```

### 2. Lancement du Backend (API)
```bash
python -m src.api.main
```

### 3. Lancement du Frontend (UI)
```bash
streamlit run src/ui/app.py
```

## Sécurité
- Ne jamais commiter le fichier `.env` contenant vos clés API.
- Utilisez `.env.example` pour documenter les variables nécessaires.
