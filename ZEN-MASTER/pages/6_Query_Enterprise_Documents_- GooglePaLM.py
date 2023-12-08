# Library Imports ###########################################################
import streamlit as st
import os
from PyPDF2 import PdfReader
import pinecone
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA


# Some important configurations to be read ##################################
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = "kedb"

# some global session variable
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

# Functions with implementation #############################################

# Function to get pdf text from all pdf docs


def get_pdf_text(pdf_docs):

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

# Create Chunks from PDF Text


def create_chunks_from_text(pdf_text):

    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=512,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_text(pdf_text)
    # chunks = text_splitter.create_text(pdf_text)
    return chunks

# Create Vector Store using Embeddings for the pdf text


def create_vectorStore_from_chunks(chunks, indexName):

    # Initialize Pinecone vector Store
    pinecone.init(
        api_key=pinecone_key,
        environment=pinecone_env
    )

    # Initialize embedings
    embeddings = GooglePalmEmbeddings()

    # create the vector store
    vector_store = Pinecone.from_texts(
        chunks, embeddings, index_name=indexName)
    # vector_store = Pinecone.from_documents(
    #    chunks, embeddings, index_name=indexName)
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

# Function to initialize Google Palm LLM


def load_llm():

    # Initialize google palm
    llm = GooglePalm(temperature=0.1)
    return llm

# Function question - answer


def question_answer():

    embeddings = GooglePalmEmbeddings()

    # Now we have to load from Pinecone index
    db = Pinecone.from_existing_index(pinecone_index_name, embeddings)

    # Now we load the llm
    llm = load_llm()

    # set the custom prompt
    qa_prompt = set_custom_prompt()

    # Establish the retreival qa chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
                                           retriever=db.as_retriever(
                                               search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': qa_prompt})

    return qa_chain


# UI Begins from here ########################################################
st.subheader("Chat with Google PaLM on a knowledge base :books:")
st.markdown(''' --- ''')

# Create tabs
tab_titles = ['Vector Store(PaLM Embeddings + Pinecone)', 'Query Google PaLM']
tabs = st.tabs(tab_titles)

with tabs[0]:

    st.subheader(
        "Create Vector Store on your KEDB using PaLM Embeddings in Pinecone")
    docs = st.file_uploader("Upload your Documents", type=[
                            'pdf'], accept_multiple_files=True)

    process_btn = st.button("Process Vector Store", key="process_btn")

with tabs[1]:

    st.subheader("Query your KEDB using Google PaLM :palm_tree: ")
    query_area = st.text_area("Enter your Query")

    response_btn = st.button("Get Response", key="response_local")

if process_btn:

    with st.spinner("Creating Pinecone Vector Store..."):

        # Get the texts from PDFs uploaded
        pdf_texts = get_pdf_text(pdf_docs=docs)
        chunks = create_chunks_from_text(pdf_text=pdf_texts)

        # create the vector store
        create_vectorStore_from_chunks(chunks, pinecone_index_name)
        st.success("Done...")

if response_btn:

    with st.spinner("Getting response from Google PaLM..."):

        if st.session_state['vectorStore'] is not None:

            result = question_answer()
            response = result({'query': query_area})
            st.write(response)
            st.success("Done...")
