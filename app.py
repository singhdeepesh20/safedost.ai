import os
import pathlib
import logging
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

# ------------------- CONFIG -------------------
load_dotenv()
DATA_DIR = pathlib.Path("data")
FAISS_DIR = pathlib.Path("./faiss_index")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("safeDost-chatbot")

# ------------------- EMBEDDINGS -------------------
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def get_embeddings():
    hf_token = os.getenv("HF_TOKEN")  # Load from Streamlit secrets
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # use L6 (smaller + stable)
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
        cache_folder=".cache",  # helps with re-use
        token=hf_token          # pass the token for auth
    )


# ------------------- LLM -------------------
def get_llm():
    groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    if not groq_key:
        st.error("❌ Please set `GROQ_API_KEY` in Streamlit secrets or .env file")
        st.stop()
    return ChatGroq(groq_api_key=groq_key, model_name="llama-3.1-8b-instant")

# ------------------- DOCS HANDLING -------------------
def load_documents():
    docs = []
    if not DATA_DIR.exists():
        st.error(f"❌ Data folder not found: {DATA_DIR}. Create it and add .txt/.pdf files.")
        st.stop()
    for file_path in sorted(DATA_DIR.glob("*")):
        if file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            continue
        docs.extend(loader.load())
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

def save_faiss(vs):
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(FAISS_DIR))

def load_faiss(embeddings):
    if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
        return FAISS.load_local(
            str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True
        )
    return None

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="SafeDost.AI Chatbot", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .chat-bubble-user {
        text-align: right;
        background-color: #ffffff;
        color: #000000;
        padding: 10px;
        border-radius: 15px 15px 0 15px;
        margin: 5px 0;
        max-width: 70%;
        float: right;
        clear: both;
        word-wrap: break-word;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .chat-bubble-bot {
        text-align: left;
        background-color: #e6e6e6;
        color: #000000;
        padding: 10px;
        border-radius: 15px 15px 15px 0;
        margin: 5px 0;
        max-width: 70%;
        float: left;
        clear: both;
        word-wrap: break-word;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ SafeDost.AI – A Digital Dost for Women’s Safety")
st.write("""
SafeDost.AI is that invisible companion — a dost that cares, protects, and empowers women every step of the way.

We dream of a future where every woman knows:  
“Even if I’m walking alone… I’m never truly alone."
""")

# ------------------- SESSION STATE -------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------- VECTORSTORE INIT -------------------
if st.session_state.vectorstore is None:
    embeddings = get_embeddings()
    vectorstore = load_faiss(embeddings)
    if vectorstore is None:
        docs = load_documents()
        if not docs:
            st.error("No documents found in data folder.")
            st.stop()
        chunks = split_docs(docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        save_faiss(vectorstore)
        st.success(f"✅ Preloaded {len(chunks)} knowledge chunks from ./data")
    st.session_state.vectorstore = vectorstore

# ------------------- CHAT HISTORY -------------------
st.subheader("💬 Chat with your Dost")
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    if role == "user":
        st.markdown(f"<div class='chat-bubble-user'>🧑‍💻 {content}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-bot'>🤖 {content}</div>", unsafe_allow_html=True)

# ------------------- CHAT INPUT -------------------
query = st.chat_input("Type your safety question here...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(query)

    # prepare context
    context_texts = []
    for i, d in enumerate(docs):
        snippet = d.page_content.strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        context_texts.append(f"Doc {i+1}:\n{snippet}\n---\n")

    user_prompt = (
        "You are SafeDost.AI assistant. Use the context to provide verified safety guidance.\n\n"
        f"Context:\n{''.join(context_texts)}\n\nUser Question: {query}\n"
        "Answer clearly, concisely, and list sources. If not present in the context, give safe general guidance."
    )

    llm = get_llm()
    with st.spinner("💡 Thinking..."):
        try:
            response = llm.invoke([HumanMessage(content=user_prompt)])
            answer = response.content
        except Exception as e:
            logger.exception("LLM generation failed: %s", e)
            answer = f"⚠️ Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.expander("📖 Sources for this answer"):
        for idx, d in enumerate(docs):
            metadata = getattr(d, "metadata", {}) or {}
            md_text = ", ".join(f"{k}:{v}" for k, v in metadata.items()) if metadata else ""
            st.markdown(f"**Source {idx+1}** {md_text}\n\n{d.page_content[:400]}...")

    st.rerun()
