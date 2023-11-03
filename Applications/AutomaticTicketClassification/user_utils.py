import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import joblib

# Function to pull index data from Pinecone
def pull_from_pinecone(pinecone_api_key, pinecone_environment, pinecone_index_name, embeddings):

    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_environment
    )

    index = Pinecone.from_existing_index(index_name=pinecone_index_name, embedding=embeddings)
    return index

# Create embeddings instance
def create_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

# This function will help us to fetch the top relevant documents from our vector store - Pinecone Index
def get_similar_docs(index, query, k=2):

    similar_docs = index.similarity_search(query, k=k)
    return similar_docs

# This function will return the fined tuned response
def get_answer(docs, query):
    chain = load_qa_chain(OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model='text-davinci-003'),
                          chain_type="stuff")
    with get_openai_callback() as ch:
        response = chain.run(input_documents=docs, question=query)
    return response

# Predict the proabble department for the ticket based on the query
def predict(query_result):
    Fitmodel = joblib.load('modelsvm.pk1')
    result = Fitmodel.predict([query_result])
    return result[0]
