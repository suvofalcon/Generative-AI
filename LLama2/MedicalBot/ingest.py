# Imports

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Get the paths
DATA_PATH = os.path.join(os.getenv("AI_DATASETS_PATH"),
                         "genai_datasets/Docs/medicalbot")

DB_FAISS_PATH = os.path.join(
    os.getenv("HOME"), "vectorstores/medicalbot/db_faiss")

print(DATA_PATH)
print(DB_FAISS_PATH)


# Create the vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)

    # Load the directory
    documents = loader.load()
    print(len(documents))

    # Now we will define the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=50)

    # Use this splitter to split the documents
    texts = text_splitter.split_documents(documents)

    # Now initialize the embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # We will use this embeddings on the texts data to create the vectorstore
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    create_vector_db()
