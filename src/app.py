"""
app.py - Interfaz de línea de comandos para el asistente médico RAG.

Uso:
    python src/app.py

Prerequisito: haber corrido ingest.py primero para crear el vector store.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from chain import create_rag_chain, ask


def main():
    print("=" * 60)
    print(" ASISTENTE MÉDICO RAG - Consenso HTA SAC 2025")
    print("=" * 60)
    print("Hacé preguntas sobre el Consenso Argentino de HTA.")
    print("Escribí 'salir' para terminar.\n")
    
    # Inicializar la chain (carga vectorstore + LLM)
    print(" Inicializando sistema...\n")
    chain, retriever = create_rag_chain()
    print("\nSistema listo!\n")
    
    while True:
        question = input("👤 Tu pregunta: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ["salir", "exit", "quit"]:
            print("\n👋 ¡Hasta luego!")
            break
        
        try:
            print("\n Buscando en el consenso...")
            response = ask(chain, retriever, question)
            
            print(f"\nRespuesta:\n{response['result']}")
            
            # Mostrar fuentes
            sources = response.get("source_documents", [])
            if sources:
                pages = set(doc.metadata.get("page", "?") for doc in sources)
                print(f"\nFuentes: páginas {', '.join(str(p) for p in sorted(pages))}")
            
            print()
        
        except Exception as e:
            print(f"\n Error: {e}\n")


if __name__ == "__main__":
    main()
