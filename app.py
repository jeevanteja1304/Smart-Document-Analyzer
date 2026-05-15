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

/* AI Cards */
.ai-box {
    background: rgba(17, 24, 39, 0.9);
    border: 1px solid #374151;
    border-radius: 18px;
    padding: 25px;
    margin-top: 20px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
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

st.markdown(
    '<div class="main-title">📄 Smart Document Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Advanced AI-Powered Multi-Document RAG Assistant</div>',
    unsafe_allow_html=True
)

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
    os.makedirs("vectorstore", exist_ok=True)

    # Save uploaded file
    file_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ File Uploaded Successfully!")

    # File Details
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

    # ---------------- READ FILE ---------------- #

    extracted_text = ""

    with st.spinner("📚 AI is analyzing the document..."):

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

        # Split Text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=250
        )

        docs = splitter.create_documents(
            [extracted_text]
        )

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector Database
        vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

        vectorstore.save_local(
            "vectorstore"
        )

    st.success("✅ Document Processed Successfully!")

    # ---------------- AI MODEL ---------------- #

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    # ---------------- DOCUMENT CONTEXT ---------------- #

    full_text = extracted_text[:7000]

    # ---------------- SUMMARY ---------------- #

    summary_prompt = f"""
You are an advanced AI document analyzer.

Analyze the uploaded document carefully and generate:

1. Detailed summary
2. Important names
3. Organizations
4. Technologies
5. Key topics
6. 5 smart suggested questions

Be accurate and concise.

DOCUMENT CONTENT:
{full_text}
"""

    with st.spinner("🤖 Generating AI Summary..."):

        try:

            summary_response = llm.invoke(
                summary_prompt
            )

            st.markdown(
                '<div class="ai-box">',
                unsafe_allow_html=True
            )

            st.subheader("📌 AI Document Summary")

            st.write(
                summary_response.content
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(f"Error: {e}")

    st.markdown("---")

    # ---------------- QUESTION SECTION ---------------- #

    question = st.text_input(
        "💬 Ask anything about the document"
    )

    ask_button = st.button(
        "🚀 Generate Answer"
    )

    # ---------------- ANSWER GENERATION ---------------- #

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

            # Greetings
            if question.lower().strip() in greetings:

                st.markdown(
                    '<div class="ai-box">',
                    unsafe_allow_html=True
                )

                st.subheader("🤖 AI Assistant")

                st.write("""
Hello 👋

I am your Smart Document Analyzer AI.

You can ask questions related to the uploaded document.

Examples:
- What is the student name?
- Summarize the document
- What company is mentioned?
- What technologies are used?
- Explain the report
""")

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                with st.spinner("🤖 Answer generating..."):

                    # Better Retrieval
                    matching_docs = vectorstore.similarity_search(
                        question,
                        k=5
                    )

                    # Retrieved Context
                    retrieved_context = "\n\n".join(
                        [
                            doc.page_content
                            for doc in matching_docs
                        ]
                    )

                    # Hybrid Context
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

                        # Generate Response
                        response = llm.invoke(
                            prompt
                        )

                        # Answer Card
                        st.markdown(
                            '<div class="ai-box">',
                            unsafe_allow_html=True
                        )

                        st.subheader("📌  Answer")

                        st.write(
                            response.content
                        )

                        st.markdown(
                            '</div>',
                            unsafe_allow_html=True
                        )

                    except Exception as e:

                        st.error(f"Error: {e}")
