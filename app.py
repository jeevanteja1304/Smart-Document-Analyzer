import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd

# PDF
from langchain_community.document_loaders import PyPDFLoader

# DOCX
from docx import Document

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Store
from langchain_community.vectorstores import FAISS

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Groq LLM
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Smart Document Analyzer",
    page_icon="📄",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: white;
    margin-top: 20px;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 40px;
    font-size: 18px;
}

/* AI BOX */
.ai-box {
    background: rgba(17, 24, 39, 0.95);
    border: 1px solid #374151;
    border-radius: 18px;
    padding: 25px;
    margin-top: 20px;
    margin-bottom: 20px;
    width: 100%;
    box-sizing: border-box;
    overflow: hidden;
}

/* CONTENT INSIDE BOX */
.ai-content {
    color: white;
    font-size: 18px;
    line-height: 1.8;
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

/* Input Box */
.stTextInput > div > div > input {
    background-color: #111827;
    color: white;
    border-radius: 12px;
    border: 1px solid #374151;
}

/* Button */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 20px;
    font-size: 16px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown("""
<div class="main-title">
📄 Smart Document Analyzer
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
Advanced AI-Powered Multi-Document RAG Assistant
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("⚙️ Features")

    st.success("✅ PDF Support")
    st.success("✅ DOCX Support")
    st.success("✅ TXT Support")
    st.success("✅ CSV Support")
    st.success("✅ AI Summary")
    st.success("✅ Suggested Questions")
    st.success("✅ Smart Question Answering")

# ---------------- FILE UPLOAD ---------------- #

uploaded_file = st.file_uploader(
    "📂 Upload Document",
    type=["pdf", "docx", "txt", "csv"]
)

# ---------------- MAIN APP ---------------- #

if uploaded_file is not None:

    # Create folders
    os.makedirs("data", exist_ok=True)

    # Save uploaded file
    file_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ File Uploaded Successfully!")

    # ---------------- FILE DETAILS ---------------- #

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "📄 File Name",
            uploaded_file.name
        )

    with col2:
        st.metric(
            "📦 File Size",
            f"{round(uploaded_file.size / 1024, 2)} KB"
        )

    # ---------------- EXTRACT TEXT ---------------- #

    extracted_text = ""

    with st.spinner("📚  analyzing the document..."):

        file_extension = uploaded_file.name.split(".")[-1].lower()

        # PDF
        if file_extension == "pdf":

            loader = PyPDFLoader(file_path)

            documents = loader.load()

            extracted_text = "\n".join(
                [doc.page_content for doc in documents]
            )

        # DOCX
        elif file_extension == "docx":

            doc = Document(file_path)

            extracted_text = "\n".join(
                [para.text for para in doc.paragraphs]
            )

        # TXT
        elif file_extension == "txt":

            with open(file_path, "r", encoding="utf-8") as f:

                extracted_text = f.read()

        # CSV
        elif file_extension == "csv":

            df = pd.read_csv(file_path)

            extracted_text = df.to_string()

        # ---------------- TEXT SPLITTING ---------------- #

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=250
        )

        docs = splitter.create_documents(
            [extracted_text]
        )

        # ---------------- EMBEDDINGS ---------------- #

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ---------------- VECTOR DATABASE ---------------- #

        vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

    st.success("✅ Document Processed Successfully!")

    # ---------------- AI MODEL ---------------- #

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    # ---------------- DOCUMENT CONTEXT ---------------- #

    full_text = extracted_text[:7000]

    # ---------------- SUMMARY PROMPT ---------------- #

    summary_prompt = f"""
You are an advanced AI document analyzer.

Analyze the uploaded document carefully and generate:

1. Detailed Summary
2. Important Names
3. Organizations
4. Technologies
5. Key Topics
6. 5 Smart Suggested Questions

Be accurate and concise.

DOCUMENT:
{full_text}
"""

    # ---------------- SUMMARY ---------------- #

    with st.spinner("🤖 Generating  Summary..."):

        try:

            summary_response = llm.invoke(
                summary_prompt
            )

            st.markdown(f"""
<div class="ai-box">

<h2>
📌  Document Summary
</h2>

<div class="ai-content">
{summary_response.content}
</div>

</div>
""", unsafe_allow_html=True)

        except Exception as e:

            st.error(f"Error: {e}")

    st.markdown("---")

    # ---------------- QUESTION INPUT ---------------- #

    question = st.text_input(
        "💬 Ask anything about the document"
    )

    ask_button = st.button(
        "🚀 Generate  Answer"
    )

    # ---------------- QUESTION ANSWERING ---------------- #

    if ask_button:

        if not question.strip():

            st.warning(
                "⚠️ Please enter a question."
            )

        else:

            greetings = [
                "hi",
                "hello",
                "hey",
                "hii",
                "hola",
                "yo"
            ]

            # ---------------- GREETING RESPONSE ---------------- #

            if question.lower().strip() in greetings:

                greeting_text = """
Hello 👋

I am your Smart Document Analyzer .

You can ask questions related to the uploaded document.

Examples:
- What is the student name?
- Summarize the document
- What company is mentioned?
- What technologies are used?
- Explain the report
"""

                st.markdown(f"""
<div class="ai-box">

<h2>
🤖 AI Assistant
</h2>

<div class="ai-content">
{greeting_text}
</div>

</div>
""", unsafe_allow_html=True)

            # ---------------- NORMAL QUESTIONS ---------------- #

            else:

                with st.spinner("🤖 generating answer..."):

                    # Similarity Search
                    matching_docs = vectorstore.similarity_search(
                        question,
                        k=5
                    )

                    # Retrieved Content
                    retrieved_context = "\n\n".join(
                        [
                            doc.page_content
                            for doc in matching_docs
                        ]
                    )

                    # Combined Context
                    context = f"""
DOCUMENT CONTENT:
{full_text}

RETRIEVED CONTENT:
{retrieved_context}
"""

                    # Prompt
                    prompt = f"""
You are an accurate AI document assistant.

Answer ONLY using the provided document content.

Rules:
- Return names and details exactly if found.
- Be accurate and concise.
- If information exists in the document, return it correctly.
- Do not hallucinate.
- If answer is unavailable, say:
'This information is not available in the document.'

DOCUMENT:
{context}

QUESTION:
{question}
"""

                    try:

                        response = llm.invoke(
                            prompt
                        )

                        st.markdown(f"""
<div class="ai-box">

<h2>
📌 AI Answer
</h2>

<div class="ai-content">
{response.content}
</div>

</div>
""", unsafe_allow_html=True)

                    except Exception as e:

                        st.error(f"Error: {e}")
