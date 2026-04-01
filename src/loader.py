"""
loader.py - Carga y chunking de documentos PDF médicos.

Este módulo se encarga de:
1. Leer PDFs usando PyMuPDF (mejor para documentos con tablas y formato complejo)
2. Dividir el texto en chunks con overlap para mantener contexto
3. Agregar metadata útil (página, fuente) a cada chunk
"""

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> list:
    """
    Carga un PDF y retorna una lista de Documents (uno por página).
    
    PyMuPDF es mejor que PyPDF para documentos médicos porque
    preserva mejor la estructura de tablas y texto formateado.
    """
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    print(f" PDF cargado: {len(documents)} páginas desde '{file_path}'")
    return documents


def split_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Divide los documentos en chunks más pequeños para el vector store.
    
    Args:
        documents: Lista de Documents de LangChain
        chunk_size: Tamaño máximo de cada chunk en caracteres
        chunk_overlap: Superposición entre chunks consecutivos.
            Esto es clave para no perder contexto en los bordes.
            Ej: si un párrafo sobre clasificación de HTA queda partido,
            el overlap asegura que la info se mantiene en al menos un chunk.
    
    Returns:
        Lista de Documents (chunks) con metadata preservada
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Separadores ordenados por prioridad: 
        # intenta cortar por doble salto de línea primero (entre secciones),
        # luego salto simple (entre párrafos), luego punto, etc.
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f" Documentos divididos en {len(chunks)} chunks")
    print(f"   Tamaño promedio: {sum(len(c.page_content) for c in chunks) // len(chunks)} caracteres")
    
    return chunks


# --- Para testear el módulo directamente ---
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python loader.py <ruta_al_pdf>")
        print("Ejemplo: python loader.py data/consenso_hta_sac_2025.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)
    
    # Mostrar los primeros 3 chunks como preview
    print("\n--- Preview de los primeros 3 chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n📄 Chunk {i+1} (página {chunk.metadata.get('page', '?')}):")
        print(chunk.page_content[:300] + "...")
        print(f"   [Longitud: {len(chunk.page_content)} chars]")
