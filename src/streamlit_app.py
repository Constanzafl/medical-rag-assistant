"""

"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from chain import create_rag_chain, ask

# --- Configuración de la página ---
st.set_page_config(
    page_title="Asistente HTA - Consenso SAC 2025",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 Asistente Médico RAG")
st.caption("Consultas sobre el Consenso Argentino de Hipertensión Arterial — SAC 2025")

# --- Inicializar chain (una sola vez, se cachea) ---
@st.cache_resource
def init_chain():
    # Si no existe el vector store, correr la ingesta
    if not os.path.exists("./chroma_db") or not os.listdir("./chroma_db"):
        from ingest import ingest
        ingest("data/COMPLETO-E-41.pdf")
    chain, retriever = create_rag_chain()
    return chain, retriever

# @st.cache_resource
# def init_chain():
#     chain, retriever = create_rag_chain()
#     return chain, retriever

# chain, retriever = init_chain()

# --- Historial de chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Input del usuario ---
if question := st.chat_input("Hacé tu pregunta sobre el consenso de HTA..."):
    # Mostrar pregunta del usuario
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Buscando en el consenso..."):
            response = ask(chain, retriever, question)

        st.markdown(response["result"])

        # Mostrar fuentes en un expander
        sources = response.get("source_documents", [])
        if sources:
            pages = sorted(set(doc.metadata.get("page", "?") for doc in sources))
            with st.expander(f"📚 Fuentes: páginas {', '.join(str(p) for p in pages)}"):
                for i, doc in enumerate(sources):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i+1} — Página {page}**")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()

    # Guardar respuesta en historial
    st.session_state.messages.append({"role": "assistant", "content": response["result"]})

# --- Sidebar ---
with st.sidebar:
    st.header("ℹ️ Sobre este asistente")
    st.markdown(
        """
        Este sistema utiliza **RAG** (Retrieval-Augmented Generation) 
        para responder preguntas basándose exclusivamente en el 
        Consenso Argentino de HTA de la SAC 2025.
        
        **Stack técnico:**
        - LLM: Llama 3 / GPT-OSS via Groq
        - Embeddings: sentence-transformers (multilingual)
        - Vector Store: ChromaDB
        - Orquestación: LangChain
        
        **Autora:** [Constanza Florio](https://www.linkedin.com/in/mariaconstanzaflorio)
        """
    )
    
    if st.button("Limpiar historial"):
        st.session_state.messages = []
        st.rerun()