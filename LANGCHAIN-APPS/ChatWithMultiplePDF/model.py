# Imports
#
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms.openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

DB_FAISS_PATH = os.path.join(os.getenv("HOME"),
                             "vectorstores/multipdf/db_faiss")

'''
Create Chunks from PDF Text
'''
def create_chunks_from_pdfText(pdf_text):
    
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    chunks = text_splitter.split_text(pdf_text)


    return chunks

'''
Create Vector Store from Chunks
'''
def create_vector_store_from_chunks(openai_key, text_chunks):

    # Initialize embeddings
    embedding = OpenAIEmbeddings(openai_api_key=openai_key)
    vector_store = FAISS.from_texts(text_chunks, embedding)
    
    vector_store.save_local(DB_FAISS_PATH)

    return vector_store


'''
Function to perform search, query LLM and return results
'''
def perform_search_return_results(key, query, vector_store):

    if query:
        docs = vector_store.similarity_search(query=query, k=3)

        # Initialize the llm
        llm = OpenAI(api_key=key)
        chain = load_qa_chain(llm=llm, chain_type='stuff')

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)

            return response
