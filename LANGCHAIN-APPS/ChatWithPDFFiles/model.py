# Imports
#
from PyPDF2 import PdfReader
from langchain.llms.openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

'''
Function to create text chunks from pdf file
'''
def create_chunks_from_pdf(pdf_file):

    pdf_reader = PdfReader(pdf_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    # define the text text_splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text=text)

    return chunks

'''
Function to create embeddings
'''
def create_embeddings_from_text(key, text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=key)

    # create the store
    vector_store = FAISS.from_texts(text_chunks,embeddings)

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

