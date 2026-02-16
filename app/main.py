import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

st.title("RAG Chatbot with PDF")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google API Key", type="password")
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    st.warning("Please provide a Google API Key to proceed.")
    st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# PDF Processing
if uploaded_file is not None and st.session_state.vector_store is None:
    with st.spinner("Processing PDF..."):
        try:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Load and chunk
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)

            # Embeddings and Vector Store
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)
            st.session_state.vector_store = vector_store

            st.success("PDF Processed Successfully!")
            os.remove(tmp_path)
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the PDF"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vector_store:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
            retriever = st.session_state.vector_store.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )

            with st.spinner("Generating answer..."):
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]

                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("Source Documents"):
                        for doc in response["source_documents"]:
                            st.write(doc.page_content)
        except Exception as e:
            st.error(f"Error generating answer: {e}")
    else:
        st.warning("Please upload a PDF first.")

