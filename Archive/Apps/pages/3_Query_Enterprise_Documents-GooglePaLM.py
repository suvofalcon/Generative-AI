# Library Imports

import streamlit as st
import os

from PyPDF2 import PdfReader
import pinecone
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA

# Read Configurations
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = "kedb"

google_genai_key = os.getenv("GOOGLE_API_KEY")

# Defining a global session variable
st.session_state['vectorStore'] = ''

# define the custom prompt template
custom_prompt_template = """Use the following pieces of information to answer
user's question. In case you dont know the answer, just say you dont know
the answer, dont try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.

Helpful answer:
"""

# Function to get text from all pdf docs
def get_pdf_text(pdf_docs):

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

# Create Chunks from Text
def create_chunks_from_text(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size = 512,
        chunk_overlap = 50,
        length_function = len
    )
    chunks = text_splitter.split_text(pdf_text)
    return chunks

# create vector store using Embeddings from pdf text
def create_vectorStore_from_chunks(chunks, indexName):

    # Initialize Pinecone vector store
    pinecone.Pinecone(
        api_key = pinecone_key,
        environment = pinecone_env
    )

    # Initialize the embeddings
    embeddings = GooglePalmEmbeddings()

    # Create the vector store
    vector_store = Pinecone.from_texts(texts=chunks, embedding=embeddings,
                                       index_name=indexName)
    # assign the session variable
    st.session_state['vectorStore'] = vector_store

    # if an index is already existing , we can load like this
    # vector_store = Pinecone.from_existing_index(indexName, embeddings)
    return vector_store

# Function to set the custom prompt
def set_custom_prompt():

    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )
    return prompt

# Function to initialise Google PaLM llm
def load_llm():

    # intialize google palm
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=google_genai_key)
    return llm

# Function Question - Answer

def question_answer():

    embeddings = GooglePalmEmbeddings()

    # Now we have to load the Pinecone index
    db = Pinecone.from_existing_index(index_name=pinecone_index_name,
                                      embedding=embeddings)

    # Now we load the llm
    llm = load_llm()

    # Set the custom prompt
    qa_prompt = set_custom_prompt()

    # Establish the Retrieval QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever = db.as_retriever(search_kwargs={'k':2}),
        return_source_documents = True,
        chain_type_kwargs = {'prompt':qa_prompt}
    )

    return qa_chain

# UI Begins from here .................
st.subheader("Chat with Google GenerativeAI on knowledge base :books:")
st.markdown(''' --- ''')

# Create tabs
tab_titles = ['VectorStore(PaLM Embeddings + PineCone)', 'Query Google GenAI']
tabs = st.tabs(tabs=tab_titles)

with tabs[0]:

    st.subheader("Create Vector Store from your Knowledge Documents using Google PaLM embeddings and on Pinecone")
    docs = st.file_uploader("Upload your Documents", type=['pdf'],
                            accept_multiple_files=True)

    process_btn = st.button("Process Vector Store", key="process_btn")

with tabs[1]:

    st.subheader("Query your Documents using Google GenerativeAI :palm_tree")
    query_area = st.text_area("Enter your Query")

    response_btn = st.button("Get Response", key="response_btn")

if process_btn:

    with st.spinner("Creating Pinecone Vector Store...."):

        if st.session_state['vectorStore'] is not None:

            # Get the texts from PDFs uploaded
            pdf_texts = get_pdf_text(pdf_docs = docs)
            chunks = create_chunks_from_text(pdf_text=pdf_texts)

            # Create the vector store
            create_vectorStore_from_chunks(chunks=chunks,indexName=pinecone_index_name)

            st.success("Done...!!")

if response_btn:

    with st.spinner("Getting response from Google GenerativeAI..."):

        # if st.session_state['vectorStore'] is not None:

        #     result = question_answer()
        #     response = result({'query': query_area})
        #     st.write(response)
        #     st.success('Done...!!')

        if (Pinecone.from_existing_index(index_name=pinecone_index_name,
                                         embedding=GooglePalmEmbeddings())) is not None:

            st.write("Pinecone index found...")
            st.markdown(" --- ")
            result = question_answer()
            response = result({'query': query_area})
            st.write(result)
            st.success("Done...!!")