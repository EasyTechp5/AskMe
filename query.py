from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

CHROMA_DIR = "/chroma"

# Load vector store and model
def load_qa_chain(model_name="phi"):
    embeddings = OllamaEmbeddings(model=model_name)
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = db.as_retriever()

    llm = Ollama(model=model_name)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
