from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATAPATH = "Data"
CHROMAPATH = "Chroma"

#load documents (PDFs) from a directory
def load_docs():
    document_loaders = PyPDFDirectoryLoader(DATAPATH, glob="*.pdf")
    documents = document_loaders.load()
    return documents

# split documents into smaller chunks of text
def split_docs(documents:list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    split_documents = text_splitter.split_documents(documents)
    return split_documents

# embed documents using OllamaEmbeddings and store in Chroma vector store
def embed_func():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def vector_store_docs(documents:list[Document], embeddings):
    vector_store = Chroma.from_documents(
        documents, 
        embeddings, 
        persist_directory=CHROMAPATH)
    return vector_store

# main function to execute the workflow
def main():
    documents = load_docs()
    print(f"Loaded {len(documents)} documents (pages).")
    
    split_documents = split_docs(documents)
    print(f"Split into {len(split_documents)} chunks.")
    
    embeddings = embed_func()
    vector_store = vector_store_docs(split_documents, embeddings)
    print("Documents embedded and stored in Chroma.")

if __name__ == "__main__":
    main()

