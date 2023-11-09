import os
from pypdf import PdfReader
import pinecone
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import Pinecone
from langchain.schema import Document, embeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


# Extract information from the PDF file
def get_pdf_text(pdf_doc):

    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


# iterate over files in a list of user uploaded pdf files one by one
def create_docs(user_pdf_list, unique_id):
    docs = []
    for filename in user_pdf_list:

        chunks = get_pdf_text(filename)

        # adding item to our list and adding metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name, "id": filename.file_id, "type": filename.type,
                      "size": filename.size, "unique_id": unique_id}
        ))

    return docs


# create embeddings instance
def create_embeddings_load_data():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-V2")
    return embeddings


# function to push data to vector store(Pinecone)
def push_to_pinecone(pinecone_apiKey, pinecone_environment, pinecone_index_name, embeddings, docs):

    pinecone.init(api_key=pinecone_apiKey,
                  environment=pinecone_environment)

    print("data pushed to pinecone")
    Pinecone.from_documents(docs, embeddings, index_name=pinecone_index_name)


# function to pull data from vecrtor store
def pull_from_pinecone(pinecone_apiKey, pinecone_environment,
                       pinecone_index_name, embeddings):

    pinecone.init(api_key=pinecone_apiKey,
                  environment=pinecone_environment)

    index = Pinecone.from_existing_index(pinecone_index_name, embeddings)
    return index


# function to help us get relevant documents
# from vector store based on user input
def similar_docs(query, number_of_resumes, pinecone_apiKey,
                 pinecone_environment, pinecone_index_name,
                 embeddings, unique_id):

    pinecone.init(api_key=pinecone_apiKey,
                  environment=pinecone_environment)

    index = pull_from_pinecone(
        pinecone_apiKey, pinecone_environment, pinecone_index_name, embeddings)

    similar_docs = index.similarity_search_with_score(
        query, int(number_of_resumes), {"unique_id": unique_id})
    return similar_docs

# Function which helps us to get the summary of a document


def get_summary(current_doc):
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary
