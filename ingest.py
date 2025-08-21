import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# Set paths
PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "FaQ_for_EasyTech.pdf")

# Load PDF text
def load_pdf_text(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found at {path}")
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Split text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# Embed and store in FAISS (no persistence issues)
def create_vector_store():
    raw_text = load_pdf_text(PDF_PATH)
    texts = split_text(raw_text)

    embeddings = OllamaEmbeddings(model="phi")  # Change model if needed

    vectorstore = FAISS.from_texts(texts, embeddings)
    print("[✔] PDF embedded and stored in FAISS (in-memory).")

    return vectorstore

if __name__ == "__main__":
    create_vector_store()
