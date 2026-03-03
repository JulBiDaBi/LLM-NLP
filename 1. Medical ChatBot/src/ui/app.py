import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Medical ChatBot", page_icon="🏥")

st.title("🏥 Medical ChatBot")
st.markdown("Posez vos questions sur la santé et la médecine.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = requests.post(f"{API_URL}/ask", json={"question": prompt})
            if response.status_code == 200:
                answer = response.json().get("answer", "Désolé, je n'ai pas pu obtenir de réponse.")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error(f"Erreur de l'API : {response.status_code}")
        except Exception as e:
            st.error(f"Erreur de connexion : {e}")
