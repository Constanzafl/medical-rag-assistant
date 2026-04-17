"""
Streamlit UI para el Agente Médico con Tool Calling
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler

import requests
import fitz  # PyMuPDF


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Agente Médico",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Agente Médico con Tool Calling")
st.caption("LangChain + Groq + PubMed + OpenFDA + PDF Analysis")


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("⚙️ Configuración")

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        st.success("✅ Groq API Key cargada desde .env")
    else:
        st.error("⚠️ Falta GROQ_API_KEY en el archivo .env")

    uploaded_pdfs = st.file_uploader(
        "📄 Subí PDFs médicos adicionales (opcional)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.session_state.get("pdf_collection"):
        st.success(f"📄 {len(st.session_state.pdf_collection)} documento(s) cargados")
        for name in st.session_state.pdf_collection:
            st.caption(f"• {name}")

    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.subheader("🔧 Tools disponibles")
    st.markdown("""
    - **🔬 PubMed** — Artículos científicos
    - **💊 OpenFDA** — Info de medicamentos
    - **📄 PDF** — Análisis de documentos
    - **❤️ Riesgo CV** — Calculadora clínica
    """)

    st.divider()
    st.caption("Stack: LangChain · Groq · PubMed API · OpenFDA API · PyMuPDF")


# =============================================================================
# PDF HANDLING — Múltiples documentos
# =============================================================================

def extract_pdf_text(source) -> str | None:
    """Extrae texto de un PDF (path o uploaded file)."""
    try:
        if isinstance(source, str):
            doc = fitz.open(source)
        else:
            doc = fitz.open(stream=source.read(), filetype="pdf")
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text)
        doc.close()
        return "\n\n".join(pages) if pages else None
    except Exception:
        return None


# Inicializar colección de PDFs
if "pdf_collection" not in st.session_state:
    st.session_state.pdf_collection = {}

# Cargar PDF default del .env (si existe y no se cargó ya)
default_pdf = os.getenv("MEDICAL_PDF_PATH")
if default_pdf and os.path.exists(default_pdf):
    pdf_name = os.path.basename(default_pdf)
    if pdf_name not in st.session_state.pdf_collection:
        text = extract_pdf_text(default_pdf)
        if text:
            st.session_state.pdf_collection[pdf_name] = text

# Cargar PDFs subidos por la UI
if uploaded_pdfs:
    for pdf_file in uploaded_pdfs:
        if pdf_file.name not in st.session_state.pdf_collection:
            text = extract_pdf_text(pdf_file)
            if text:
                st.session_state.pdf_collection[pdf_file.name] = text


# =============================================================================
# TOOLS
# =============================================================================

@tool
def buscar_pubmed(query: str, max_resultados: int = 3) -> str:
    """Busca artículos científicos en PubMed.
    Usar cuando el usuario pregunta por evidencia, estudios, o literatura médica.
    La query debe estar en inglés para mejores resultados."""

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    try:
        search_resp = requests.get(
            f"{base_url}/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": max_resultados, "retmode": "json", "sort": "relevance"},
            timeout=10,
        )
        search_resp.raise_for_status()
        ids = search_resp.json()["esearchresult"]["idlist"]

        if not ids:
            return f"No se encontraron artículos para: '{query}'"

        summary_resp = requests.get(
            f"{base_url}/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=10,
        )
        summary_resp.raise_for_status()
        results = summary_resp.json()["result"]

        articulos = []
        for uid in ids:
            if uid in results:
                art = results[uid]
                titulo = art.get("title", "Sin título")
                autores = art.get("authors", [])
                autor_str = autores[0]["name"] + " et al." if autores else "Desconocido"
                fecha = art.get("pubdate", "?")
                revista = art.get("fulljournalname", art.get("source", ""))
                articulos.append(
                    f"- {titulo}\n  {autor_str} | {revista} | {fecha}\n"
                    f"  https://pubmed.ncbi.nlm.nih.gov/{uid}/"
                )

        return f"{len(articulos)} artículos encontrados:\n\n" + "\n\n".join(articulos)

    except requests.RequestException as e:
        return f"Error al consultar PubMed: {e}"


@tool
def buscar_medicamento(nombre_droga: str) -> str:
    """Busca información sobre un medicamento en OpenFDA.
    Devuelve indicaciones, advertencias y efectos adversos.
    El nombre debe estar en inglés (ej: 'metformin', 'enalapril')."""

    try:
        resp = requests.get(
            "https://api.fda.gov/drug/label.json",
            params={"search": f'openfda.generic_name:"{nombre_droga}"', "limit": 1},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("results"):
            return f"No se encontró información para '{nombre_droga}'."

        drug = data["results"][0]

        def extraer(campo: str, max_chars: int = 500) -> str:
            texto = drug.get(campo, ["No disponible"])
            contenido = texto[0] if isinstance(texto, list) else str(texto)
            return contenido[:max_chars] + "..." if len(contenido) > max_chars else contenido

        nombre = drug.get("openfda", {}).get("brand_name", ["Desconocido"])
        nombre_str = nombre[0] if isinstance(nombre, list) else nombre

        return (
            f"=== {nombre_droga.upper()} (Marca: {nombre_str}) ===\n\n"
            f"INDICACIONES:\n{extraer('indications_and_usage')}\n\n"
            f"ADVERTENCIAS:\n{extraer('warnings')}\n\n"
            f"EFECTOS ADVERSOS:\n{extraer('adverse_reactions')}"
        )

    except requests.RequestException as e:
        return f"Error al consultar OpenFDA: {e}"


@tool
def analizar_pdf(pregunta: str) -> str:
    """Busca información en los documentos PDF, guías clínicas y consensos cargados.
    Usar cuando el usuario menciona: guía, documento, consenso, protocolo,
    o pregunta sobre recomendaciones clínicas que podrían estar en los PDFs cargados.
    SIEMPRE usar esta tool antes que PubMed si la pregunta parece referirse a un documento local."""

    collection = st.session_state.get("pdf_collection", {})
    if not collection:
        return "No hay PDFs cargados. Pedile al usuario que suba un documento."

    keywords = pregunta.lower().split()
    stopwords = {"qué", "que", "cuál", "cual", "cómo", "como", "el", "la",
                  "los", "las", "de", "del", "en", "un", "una", "es", "son",
                  "para", "por", "con", "se", "al", "y", "o", "a"}
    keywords = [k for k in keywords if k not in stopwords and len(k) > 2]

    all_results = []
    for doc_name, full_text in collection.items():
        parrafos = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 50]
        for p in parrafos:
            p_lower = p.lower()
            score = sum(1 for kw in keywords if kw in p_lower)
            if score > 0:
                all_results.append((score, doc_name, p))

    all_results.sort(key=lambda x: x[0], reverse=True)

    if not all_results:
        return f"No se encontró contenido relevante para: '{pregunta}'"

    fragmentos = []
    for score, doc_name, texto in all_results[:3]:
        fragmentos.append(f"[{doc_name}] (relevancia: {score}):\n{texto[:1000]}")

    return (
        f"Contenido relevante de {len(collection)} documento(s):\n\n"
        + "\n\n---\n\n".join(fragmentos)
    )


@tool
def calcular_riesgo_cv(
    edad: int,
    colesterol_total: int,
    presion_sistolica: int,
    fumador: bool,
    diabetes: bool,
) -> str:
    """Calcula un score simplificado de riesgo cardiovascular.
    Usar cuando el usuario da datos clínicos y pregunta por riesgo CV."""
    score = 0
    if edad > 55: score += 2
    elif edad > 45: score += 1
    if colesterol_total > 240: score += 2
    elif colesterol_total > 200: score += 1
    if presion_sistolica > 140: score += 2
    elif presion_sistolica > 120: score += 1
    if fumador: score += 2
    if diabetes: score += 2

    if score >= 6:
        nivel, rec = "ALTO", "Derivar a cardiología. Considerar inicio de estatinas."
    elif score >= 3:
        nivel, rec = "MODERADO", "Cambios en estilo de vida. Reevaluar en 3-6 meses."
    else:
        nivel, rec = "BAJO", "Mantener controles anuales y hábitos saludables."

    return f"Score: {score}/10 — Riesgo: {nivel}\nRecomendación: {rec}"


# =============================================================================
# CALLBACK — Razonamiento del agente
# =============================================================================

class StreamlitToolCallbackHandler(BaseCallbackHandler):
    """Muestra en Streamlit el razonamiento del agente."""

    def __init__(self, container):
        self.container = container
        self.steps = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "tool")
        emoji_map = {
            "buscar_pubmed": "🔬",
            "buscar_medicamento": "💊",
            "analizar_pdf": "📄",
            "calcular_riesgo_cv": "❤️",
        }
        emoji = emoji_map.get(tool_name, "🔧")
        self.container.info(f"{emoji} Usando **{tool_name}**...")
        self.steps.append({"tool": tool_name, "emoji": emoji})

    def on_tool_end(self, output, **kwargs):
        if self.steps:
            self.steps[-1]["output"] = output

    def show_reasoning(self):
        if self.steps:
            with self.container.expander("🧠 Ver razonamiento del agente"):
                for i, step in enumerate(self.steps, 1):
                    st.markdown(f"**Paso {i}: {step['emoji']} {step['tool']}**")
                    if "output" in step:
                        st.code(step["output"][:1000], language=None)
                    if i < len(self.steps):
                        st.caption("↓ El agente decidió que necesitaba más info...")


# =============================================================================
# AGENT SETUP
# =============================================================================

@st.cache_resource
def get_agent_executor(_groq_key: str):
    """Crea el agente (cacheado para no recrearlo en cada interacción)."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=_groq_key)
    tools = [buscar_pubmed, buscar_medicamento, analizar_pdf, calcular_riesgo_cv]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Sos un asistente médico avanzado para profesionales de la salud.\n"
            "Tenés acceso a estas herramientas:\n"
            "- analizar_pdf: para consultar guías clínicas, consensos y documentos cargados. "
            "PRIORIZÁ esta tool cuando el usuario mencione 'guía', 'consenso', 'documento' o 'protocolo'.\n"
            "- buscar_pubmed: para buscar artículos científicos en PubMed\n"
            "- buscar_medicamento: para info de drogas en OpenFDA (nombres en inglés)\n"
            "- calcular_riesgo_cv: para evaluar riesgo cardiovascular\n\n"
            "Podés combinar múltiples herramientas si es necesario.\n"
            "Respondé en español. Citá fuentes cuando uses PubMed o OpenFDA.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=8,
        handle_parsing_errors=True,
    )


# =============================================================================
# CHAT UI
# =============================================================================

# Historial de mensajes (para la UI de Streamlit)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Historial para el agente (en formato LangChain)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input del usuario
if prompt := st.chat_input("Hacé una pregunta médica..."):

    if not groq_key:
        st.error("⚠️ Falta GROQ_API_KEY en el archivo .env")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        tool_container = st.container()
        callback = StreamlitToolCallbackHandler(tool_container)

        try:
            agent_executor = get_agent_executor(groq_key)
            result = agent_executor.invoke(
                {
                    "input": prompt,
                    "chat_history": st.session_state.chat_history,
                },
                config={"callbacks": [callback]},
            )
            response = result["output"]
        except Exception as e:
            response = f"Error: {str(e)}"

        callback.show_reasoning()
        st.markdown(response)

        # Guardar en ambos historiales
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=response))

        # Limitar memoria para no exceder contexto del LLM
        MAX_HISTORY = 20
        if len(st.session_state.chat_history) > MAX_HISTORY:
            st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]