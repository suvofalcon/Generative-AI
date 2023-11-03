from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import pinecone
import asyncio
from langchain.document_loaders.sitemap import SitemapLoader

# Function to fetch data from website
# https://python.langchain.com/docs/modules/data_connection/document_loaders/integrations/sitemap
# SitemapLoader loads a sitemap from a given URL, and then scrape and load all pages in the sitemap, returning each page as a Document.

def get_website_data(siteman_url):

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loader = SitemapLoader(siteman_url)

    docs = loader.load()
    return docs


# Function to split the data into smaller chunks
def split_data(docs):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs_chunks = text_splitter.split_documents(docs)
    return docs_chunks

# Function to create embeddings instance
def create_embeddings():

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") # We can use OpenAI embeddings also, but we are using an Open Source embedding
    return embeddings

# Push Data to Pinecone
def push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs_chunks):

    pinecone.init(
        api_key=pinecone_apikey,
        environment=pinecone_environment
    )

    index = Pinecone.from_documents(docs_chunks, embedding=embeddings, index_name=pinecone_index_name)
    return index

# Function to pull index data from Pinecone
def pull_from_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings):

    pinecone.init(
        api_key=pinecone_apikey,
        environment=pinecone_environment
    )

    index = Pinecone.from_existing_index(pinecone_index_name, embedding=embeddings)
    return index

# This function will help us in fetching the relevant documents from our vector store - Pinecone index
def get_similar_docs(index, query, k=2):

    similar_docs = index.similarity_search(query, k=k)
    return similar_docs
