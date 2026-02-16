# RAG Assignment - PDF Chatbot

This project implements a Retrieval Augmented Generation (RAG) system that allows users to upload a PDF and ask questions based on its content.

## Features
- **PDF Text Extraction**: Uses `pypdf` to extract text from PDF documents.
- **Chunking**: Splits text into manageable chunks using `RecursiveCharacterTextSplitter`.
- **Embeddings**: Uses `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) for local, efficient vector representation.
- **Vector Store**: Uses `FAISS` for fast similarity search.
- **LLM**: Integrates with **Google Gemini** for high-quality answer generation.
- **Interfaces**:
    - **Jupyter Notebook**: Step-by-step implementation (`notebooks/rag_pipeline.ipynb`).
    - **Streamlit App**: Interactive web UI (`app/main.py`).

## Prerequisites
- Python 3.8+
- Google API Key (for Gemini)

## Setup

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Key**:
    - Create a `.env` file in the root directory.
    - Add your Google API Key:
      ```
      GOOGLE_API_KEY=your_api_key_here
      ```

## Usage

### Running the Jupyter Notebook
1.  Navigate to the `notebooks` directory.
2.  Start Jupyter:
    ```bash
    jupyter notebook
    ```
3.  Open `rag_pipeline.ipynb` and run the cells.

### Running the Streamlit App
1.  Run the app from the root directory:
    ```bash
    streamlit run app/main.py
    ```
2.  Open the provided URL in your browser.
3.  Upload a PDF and start chatting!

## Project Structure
- `data/`: Stores sample data.
- `notebooks/`: Contains the RAG pipeline notebook.
- `app/`: Contains the Streamlit application.
- `requirements.txt`: Project dependencies.