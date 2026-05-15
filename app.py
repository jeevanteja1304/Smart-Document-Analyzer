import streamlit as st
from dotenv import load_dotenv
import os

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

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

# Streamlit Config
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
    '<div class="sub-title">Advanced AI-Powered RAG PDF Assistant</div>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("⚙️ Features")

    st.success("✅ PDF Upload")
    st.success("✅ AI Summary")
    st.success("✅ Suggested Questions")
    st.success("✅ Smart Question Answering")

# ---------------- FILE UPLOAD ---------------- #

uploaded_file = st.file_uploader(
    "📂 Upload your PDF",
    type="pdf"
)

# ---------------- MAIN APP ---------------- #

if uploaded_file is not None:

    # Create folders
    os.makedirs("data", exist_ok=True)
    os.makedirs("vectorstore", exist_ok=True)

    # Save uploaded PDF
    pdf_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF Uploaded Successfully!")

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

    # ---------------- LOAD PDF ---------------- #

    with st.spinner("📚 AI is analyzing the document..."):

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        # Better Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=250
        )

        docs = splitter.split_documents(
            documents
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

    # Optimized Context
    full_text = "\n".join(
        [
            doc.page_content
            for doc in docs[:12]
        ]
    )

    # ---------------- SUMMARY ---------------- #

    summary_prompt = f"""
You are an advanced AI document analyzer.

Analyze the uploaded PDF carefully and generate:

1. Detailed summary
2. Important names
3. Organizations
4. Technologies
5. Key topics
6. Suggested questions users can ask

Be accurate and concise.

PDF Content:
{full_text}
"""

    with st.spinner("🤖 Generating  Summary..."):

        try:

            summary_response = llm.invoke(
                summary_prompt
            )

            st.markdown(
                """
                <div class="ai-box">
                """,
                unsafe_allow_html=True
            )

            st.subheader("📌 AI Document Summary")

            st.write(
                summary_response.content
            )

            st.markdown(
                """
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(f"Error: {e}")

    st.markdown("---")

    # ---------------- QUESTION INPUT ---------------- #

    question = st.text_input(
        "💬 Ask anything about the PDF"
    )

    ask_button = st.button(
        "🚀 Generate  Answer"
    )

    # ---------------- ANSWER GENERATION ---------------- #

    if ask_button:

        # Empty Question
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

            # ---------------- GREETING HANDLING ---------------- #

            if question.lower().strip() in greetings:

                st.markdown(
                    """
                    <div class="ai-box">
                    """,
                    unsafe_allow_html=True
                )

                st.subheader("🤖 AI Assistant")

                st.write("""
Hello 👋

I am your Smart Document Analyzer AI.

Please ask questions related to the uploaded PDF.

Examples:
- What is the student name?
- What company is mentioned?
- Summarize the document
- What technologies are used?
- Explain the internship report
""")

                st.markdown(
                    """
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ---------------- QUESTION ANSWERING ---------------- #

            else:

                with st.spinner("🤖 AI is generating answer..."):

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
FULL DOCUMENT CONTEXT:
{full_text}

RETRIEVED CONTENT:
{retrieved_context}
"""

                    # AI Prompt
                    prompt = f"""
You are an accurate AI PDF assistant.

Answer ONLY using the provided PDF content.

Rules:
- Return names and details exactly if found.
- Be accurate and concise.
- If information exists in the document, return it correctly.
- Do not hallucinate.
- If answer is unavailable, say:
'This information is not available in the PDF.'

PDF Content:
{context}

Question:
{question}
"""

                    try:

                        # Generate Response
                        response = llm.invoke(
                            prompt
                        )

                        # AI Answer Card
                        st.markdown(
                            """
                            <div class="ai-box">
                            """,
                            unsafe_allow_html=True
                        )

                        st.subheader("📌  Answer")

                        st.write(
                            response.content
                        )

                        st.markdown(
                            """
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    except Exception as e:

                        st.error(f"Error: {e}")