import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Set paths
PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "FaQ_for_EasyTech.pdf")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

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

# Embed and store in Chroma
def create_vector_store():
    raw_text = load_pdf_text(PDF_PATH)
    texts = split_text(raw_text)

    embeddings = OllamaEmbeddings(model="phi")  # Change model if needed

    # Ensure persistence directory exists
    os.makedirs(CHROMA_DIR, exist_ok=True)

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    print("[✔] PDF embedded and stored in Chroma.")

if __name__ == "__main__":
    create_vector_store()
