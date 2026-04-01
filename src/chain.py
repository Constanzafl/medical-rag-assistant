"""
chain.py - RAG chain: Retrieval + Generation con Groq (Llama 3).
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vectorstore import load_vectorstore

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

RAG_PROMPT_TEMPLATE = """Sos un asistente médico especializado que responde preguntas 
basándose ÚNICAMENTE en el siguiente contexto extraído del Consenso Argentino de 
Hipertensión Arterial de la Sociedad Argentina de Cardiología (SAC) 2025.

Reglas:
- Respondé SOLO con información presente en el contexto proporcionado.
- Si la información no está en el contexto, decí claramente: "No encontré esa información 
  en el documento del consenso."
- Citá la página cuando sea posible.
- Usá lenguaje médico apropiado pero accesible.
- Si hay tablas o clasificaciones relevantes, presentalas de forma clara.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""


def format_docs(docs):
    """Formatea los documentos recuperados en un solo string."""
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Página {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def get_llm():
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=1024,
    )
    print(f"✅ LLM inicializado: {GROQ_MODEL} via Groq")
    return llm


def create_rag_chain(vectorstore=None):
    if vectorstore is None:
        vectorstore = load_vectorstore()

    llm = get_llm()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG chain creada")
    return chain, retriever


def ask(chain, retriever, question: str) -> dict:
    result = chain.invoke(question)
    source_docs = retriever.invoke(question)
    return {"result": result, "source_documents": source_docs}


if __name__ == "__main__":
    chain, retriever = create_rag_chain()

    test_questions = [
        "¿Cómo se clasifica la hipertensión arterial según el consenso?",
        "¿Cuáles son los valores objetivo de presión arterial para el tratamiento?",
        "¿Qué fármacos se recomiendan como primera línea de tratamiento?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f" Pregunta: {q}")
        print(f"{'='*60}")
        response = ask(chain, retriever, q)
        print(f"\n Respuesta:\n{response['result']}")
        print(f"\n Fuentes ({len(response['source_documents'])} chunks):")
        for i, doc in enumerate(response['source_documents']):
            print(f"   - Chunk {i+1}: página {doc.metadata.get('page', '?')}")