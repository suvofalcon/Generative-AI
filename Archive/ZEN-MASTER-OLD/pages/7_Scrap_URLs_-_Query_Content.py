# Library Imports ########################################
import streamlit as st
import os
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS


# Some configurartions, constants and global variables ###
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FAISS_DB_PATH = os.path.join(os.getenv("HOME"), "vectorstores/url_data")
main_placeholder = st.empty()  # a ongoing status message variable

# Utility functions #####################################

# function to return URL data


def return_url_data(combinedURLs):

    # initialize the UnstructuredURLLoader
    loader = UnstructuredURLLoader(urls=combinedURLs)
    main_placeholder.text("Data Loading Started...✅✅✅")
    data = loader.load()

    return data


# Function to do the splits
def perform_content_split(url_data):

    # Split the data
    main_placeholder.text("Text Splitting Started - ✅✅✅")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )

    docs = text_splitter.split_documents(url_data)

    return docs

# function to create vector store


def create_vector_store(split_data):

    # create embeddings
    main_placeholder.text("Started Creating Vector Store - ✅✅✅")
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(split_data, embeddings)

    vectorStore.save_local(FAISS_DB_PATH)


# UI Starts from here ###################################
st.header("Scrap URLs and Query Content :robot_face:")

tab_titles = ['Process URLs', 'Query Content']
tabs = st.tabs(tab_titles)

urls = []  # All urls combined list

with tabs[0]:

    for i in range(3):
        url = st.text_input(f"URL {i + 1}")
        urls.append(url)

    process_btn = st.button("Process URLs")

with tabs[1]:

    query = st.text_input("Enter Your Query")

    response_btn = st.button("Get Response")

if process_btn:

    if urls:  # if the list is not empty

        data = return_url_data(combinedURLs=urls)
        docs = perform_content_split(url_data=data)
        create_vector_store(split_data=docs)

        st.success("Vector Store creation complete...")
