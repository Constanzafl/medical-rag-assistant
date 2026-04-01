"""
vectorstore.py - Embeddings y almacenamiento vectorial con ChromaDB.

Este módulo se encarga de:
1. Generar embeddings usando sentence-transformers (local, gratis)
2. Almacenar los vectores en ChromaDB (persistente en disco)
3. Proveer funciones de búsqueda por similaridad
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Modelo multilingual: funciona bien con español (el consenso SAC)
# y también con queries en inglés si las hacés bilingüe
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
PERSIST_DIRECTORY = "./chroma_db"


def get_embeddings():
    """
    Inicializa el modelo de embeddings.
    
    Usamos paraphrase-multilingual-MiniLM-L12-v2 porque:
    - Es multilingual (el consenso está en español)
    - Es liviano (~120MB) y corre local sin GPU
    - Buena calidad para retrieval
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # Cambiar a "cuda" si tenés GPU
        encode_kwargs={"normalize_embeddings": True},  # Normalizar para cosine similarity
    )
    print(f"Modelo de embeddings cargado: {EMBEDDING_MODEL}")
    return embeddings


def create_vectorstore(chunks: list, persist_directory: str = PERSIST_DIRECTORY) -> Chroma:
    """
    Crea un vector store nuevo a partir de una lista de chunks.
    Persiste en disco para no tener que re-procesar cada vez.
    
    Args:
        chunks: Lista de Documents (output de loader.split_documents)
        persist_directory: Carpeta donde se guarda ChromaDB
    
    Returns:
        Instancia de Chroma lista para queries
    """
    embeddings = get_embeddings()
    print(f" Generando embeddings para {len(chunks)} chunks... ")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="medical_docs",
    )
    
    print(f"Vector store creado con {len(chunks)} documentos en '{persist_directory}'")
    return vectorstore


def load_vectorstore(persist_directory: str = PERSIST_DIRECTORY) -> Chroma:
    """
    Carga un vector store existente desde disco.
    Útil para no re-procesar el PDF cada vez que corrés la app.
    """
    embeddings = get_embeddings()
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="medical_docs",
    )
    
    count = vectorstore._collection.count()
    print(f"Vector store cargado desde '{persist_directory}' ({count} documentos)")
    return vectorstore


def similarity_search(vectorstore: Chroma, query: str, k: int = 4) -> list:
    """
    Busca los k chunks más similares a la query.
    
    Args:
        vectorstore: Instancia de Chroma
        query: Pregunta del usuario
        k: Cantidad de chunks a retornar
    
    Returns:
        Lista de Documents relevantes
    """
    results = vectorstore.similarity_search(query, k=k)
    return results


# --- Para testear el módulo directamente ---
if __name__ == "__main__":
    # Test: cargar vectorstore existente y hacer una búsqueda
    try:
        vs = load_vectorstore()
        query = "¿Cuál es la clasificación de hipertensión arterial?"
        results = similarity_search(vs, query)
        
        print(f"\n🔍 Query: '{query}'")
        print(f"   Resultados: {len(results)} chunks\n")
        
        for i, doc in enumerate(results):
            print(f"📄 Resultado {i+1} (página {doc.metadata.get('page', '?')}):")
            print(doc.page_content[:200] + "...")
            print()
    except Exception as e:
        print(f"⚠️  Error: {e}")
        print("   Primero creá el vectorstore corriendo: python src/ingest.py")
