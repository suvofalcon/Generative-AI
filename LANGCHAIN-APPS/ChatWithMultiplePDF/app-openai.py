
# Library imports...

import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms.openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Read the important configurations
openai_key = os.getenv("OPENAI_API_KEY")
DB_FAISS_PATH = os.path.join(os.getenv("HOME"),
                             "vectorstores/multipdf/db_faiss")


# Function to get pdftext from all the uploaded PDF Files
def get_pdf_text(pdf_docs):

    text = ""
    for pdf in pdf_docs: # read every pdf in the list uploaded
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages: # for every page in the pdf being read
            text += page.extract_text()

    return text


# Create chunks from PDF text
def create_chunks_from_pdfText(pdf_text):
    
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)

    chunks = text_splitter.split_text(pdf_text)
    return chunks


# Create Vector Store from chunks
def create_vector_store_from_chunks(openai_key, text_chunks):

    # Initialize embeddings
    embedding = OpenAIEmbeddings(openai_api_key=openai_key)
    vector_store = FAISS.from_texts(text_chunks, embedding)
    
    vector_store.save_local(DB_FAISS_PATH)

    return vector_store


# Function to perform search, query LLM and return results
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

def main():

    st.header("Chat with Multiple PDF Files :books:")

    text_input = st.text_input("Enter your Query")
    response_btn = st.button("Generate Response")
    
    if "conversation" not in st.session_state:
        st.session_state['conversation'] = ''

    if "vectorStore" not in st.session_state:
        st.session_state['vectorStore'] = ''

    with st.sidebar:
        st.header("Chat with PDF - OpenAI")
        st.header("LLM Chatapp using LangChain and OpenAI")
        st.header("Create Vector Store on your Documents...")

        pdf_docs = st.file_uploader("Upload the PDF Files here and click on 'Process'", type='pdf',
                                    accept_multiple_files=True)
        process_button = st.button("Process")

        if process_button:
            with st.spinner("Processing"):

                # Extract Text from PdfReader
                pdf_text = get_pdf_text(pdf_docs=pdf_docs)

                chunks = create_chunks_from_pdfText(pdf_text=pdf_text)
                
                vector_store = create_vector_store_from_chunks(openai_key, chunks)

                st.write(f"Total Number of Chunks - **{len(chunks)}**")
                st.session_state['vectorStore'] = vector_store
                st.session_state['conversation'] = "processed"

                st.success("Done!!")

    # Get the respone by button            
    if response_btn:

        if st.session_state['conversation'] == "processed":
            # We will the search on the store and get the response from LLM
            response = perform_search_return_results(key=openai_key, query=text_input, vector_store=st.session_state['vectorStore'])
            
            st.write(response)
            
        

if __name__ == "__main__":
    main()

