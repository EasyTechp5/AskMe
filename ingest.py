import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_ollama import OllamaEmbeddings

# Set paths
PDF_PATH = "data/FaQ_for_EasyTech.pdf"
CHROMA_DIR = "/chroma"

# Load PDF text
def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Split text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500
    )
    return splitter.split_text(text)

# Embed and store in Chroma
def create_vector_store():
    raw_text = load_pdf_text(PDF_PATH)
    texts = split_text(raw_text)
    embeddings = OllamaEmbeddings(model="phi")  # Change model here if needed
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    print("[✔] PDF embedded and stored in Chroma.")

if __name__ == "__main__":
    create_vector_store()
