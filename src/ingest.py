"""
ingest.py - Script de ingesta: procesa el PDF y crea el vector store.

Este script se corre UNA VEZ (o cada vez que cambiás el documento).
Después, la app y el chain cargan el vectorstore desde disco.

Uso:
    python src/ingest.py data/consenso_hta_sac_2025.pdf
"""

import sys
import os

# Agregar src al path para imports
sys.path.insert(0, os.path.dirname(__file__))

from loader import load_pdf, split_documents
from vectorstore import create_vectorstore


def ingest(pdf_path: str):
    """Pipeline completo de ingesta."""
    
    print("=" * 50)
    print(" INGESTA DE DOCUMENTO")
    print("=" * 50)
    
    # 1. Cargar PDF
    print("\n Paso 1: Cargando PDF...")
    documents = load_pdf(pdf_path)
    
    # 2. Dividir en chunks
    print("\n Paso 2: Dividiendo en chunks...")
    chunks = split_documents(
        documents,
        chunk_size=1000,    # Experimentá con estos valores
        chunk_overlap=200,  # Si los chunks se ven cortados, aumentá overlap
    )
    
    # 3. Crear vector store
    print("\n Paso 3: Creando vector store con embeddings...")
    vectorstore = create_vectorstore(chunks)
    
    print("\n" + "=" * 50)
    print(" INGESTA COMPLETA")
    print(f"   Páginas procesadas: {len(documents)}")
    print(f"   Chunks generados: {len(chunks)}")
    print(f"   Vector store guardado en: ./chroma_db/")
    print("=" * 50)
    
    return vectorstore


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python src/ingest.py <ruta_al_pdf>")
        print("Ejemplo: python src/ingest.py data/consenso_hta_sac_2025.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: No se encontró el archivo '{pdf_path}'")
        sys.exit(1)
    
    ingest(pdf_path)
