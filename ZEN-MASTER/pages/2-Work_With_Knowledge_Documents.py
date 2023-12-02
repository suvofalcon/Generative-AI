# Library imports

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
DB_OPENAI_FAISS_PATH = os.path.join(os.getenv("HOME"),
                             "vectorstores/multipdf/db_faiss_openai")
DB_HF_FAISS_PATH = os.path.join(os.getenv("HOME"),
                             "vectorstores/multipdf/db_faiss_hf")

st.header("Chat with Knowledge Documents :open_file_folder:")
st.subheader("Use the power of LLM to query your Knowledge Base")

if "vectorStore" not in st.session_state:
    st.session_state['vectorStore'] = ''

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
    
    vector_store.save_local(DB_OPENAI_FAISS_PATH)
    st.session_state['vectorStore'] = vector_store

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


# Create tabs
tab_titles = ['Create Vector Store', 'Query LLM']
tabs = st.tabs(tab_titles)

# Add contents to each tab

with tabs[0]:
    st.subheader("Create Vector Store using Embeddings on your documents")
    
    pdf_docs = st.file_uploader("Upload the PDF Files here and click on 'Process'", type='pdf',
                                    accept_multiple_files=True)

    col1, col2 = st.columns([1, 4.0])

    with col1:
        process_openai = st.button("Process-OpenAI Embeddings")

    with col2:
        process_MiniLM = st.button("Process-HF Embeddings")

    if process_openai:
        with st.spinner("Processing OpenAI Embeddings..."):
            
            # Extract text from pdf documents
            pdf_text = get_pdf_text(pdf_docs)

            chunks = create_chunks_from_pdfText(pdf_text)
            
            vector_store = create_vector_store_from_chunks(openai_key, text_chunks=chunks)
            st.write(f"Total Number of Chunks - **{len(chunks)}**")

            st.success("Done...!")

    if process_MiniLM:
        with st.spinner("Processing HuggingFace Mini-LM-L6-V2 Embeddings..."):

            st.success("Done...!")


with tabs[1]:
    st.subheader("Query LLM on your Knowledge Base")

    text_area = st.text_area("Enter your Query")

    col1, col2 = st.columns([1, 4.0])

    with col1:
        query_openai = st.button("Query OpenAI")

    with col2:
        query_llama2 = st.button("Query LLama2")

    if query_openai:
        # We will the search on the store and get the response from LLM
        with st.spinner("Processing OpenAI Query..."):
            response = perform_search_return_results(key=openai_key, query=text_area, vector_store=st.session_state['vectorStore'])
            
            st.write(response)
 

    if query_llama2:
        st.write("query llama2")

