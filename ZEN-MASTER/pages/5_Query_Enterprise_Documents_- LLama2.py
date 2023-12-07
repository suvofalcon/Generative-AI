# Library Imports ###########################################################################################
import streamlit as st
import os
from PyPDF2 import PdfReader

from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms.ctransformers import CTransformers
from langchain.chains import RetrievalQA

# Some important configurations getting read ###############################################################
DB_FAISS_PATH = os.path.join(os.getenv("HOME"),
                             'vectorstores/kedb/db_faiss')

MODEL_PATH = os.path.join(
    os.getenv("HOME"), "models/llama-2-7b-chat.ggmlv3.q8_0.bin")

# Some global session variables
if 'vectorStore' not in st.session_state:
    st.session_state['vectorStore'] = ''

# define the custom prompt template
custom_prompt_template = """Use the following pieces of information to answer user's question.
In case you dont know the answer, just say you dont know the answer, dont try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.

Helpful answer:
"""

# Llama2 Specific Implementation Functions #########################################################

# Function to get pdftext from all the uploaded PDF Files
def get_pdf_text(pdf_docs):

    text = ""
    for pdf in pdf_docs: # read every pdf in the list uploaded
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages: # for every page in the pdf being read
            text += page.extract_text()

    return text

# Create Chunks from PDF Text
def create_chunks_from_text(pdf_text):

    text_splitter = RecursiveCharacterTextSplitter(separators="\n",
                                                   chunk_size=512,
                                                   chunk_overlap=20,
                                                   length_function=len)

    chunks = text_splitter.split_text(pdf_text)
    return chunks

# Create Vector Store from Chunks
def create_vector_store_from_chunks(text_chunks):

    # We will initialize the embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                          model_kwargs={'device': 'cpu'})

    # We will use this embeddings on the texts data to create the vectorstore
    db = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    db.save_local(DB_FAISS_PATH)
    # Set the session variable
    st.session_state['vectorStore'] = db

    return db

# Function to set the custom prompt
def set_custom_prompt():

    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )
    return prompt

# Function to load the llama2 LLM
def load_llm():

    # Load the locally downloaded model
    llm = CTransformers(model=MODEL_PATH,
                        model_type="llama",
                        max_new_tokens=512,
                        temperature=0.5)
    return llm

# question - answer
def question_answer():

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    # Now we have to load from faiss
    # either we load from local or we get from the global session variable
    # db = st.session_state['vectorStore']
    db = FAISS.load_local(DB_FAISS_PATH, embeddings=embeddings)

    # Now we load the llm
    llm = load_llm()

    # set the custom prompt
    qa_prompt = set_custom_prompt()

    # Establish the retreival qa chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': qa_prompt})

    return qa_chain

# Common UI Elements ################################################################################
st.subheader("Chat with LLama2 on a knowledge base :llama:")
st.markdown(''' --- ''')

# Create tabs
tab_titles = ['Vector Store (HF Embeddings + FAISS)', 'Query Llama2 :llama:']

tabs = st.tabs(tab_titles)

with tabs[0]:

    st.subheader("Create Vector on your KEDB using Hugging Face - all-miniLM-L6-v2 Embeddings")
    docs = st.file_uploader("Upload your Documents", type=['pdf'], accept_multiple_files=True)

    process_btn = st.button("Process Vector Store", key="process_btn")

with tabs[1]:

    st.subheader("Query your KEDB using LLama2 :llama:")
    query_area = st.text_area("Enter your Query")

    response_btn = st.button("Get Response", key="response_local")

if process_btn:
    with st.spinner("Creating Vector Store..."):

        # Get the texts from PDFs uploaded
        pdf_texts = get_pdf_text(pdf_docs=docs)
        chunks = create_chunks_from_text(pdf_text=pdf_texts)

        # create the vector store
        create_vector_store_from_chunks(text_chunks=chunks)
        st.success("Done...")

if response_btn:

    with st.spinner('Getting Response from Llama2...'):
        if st.session_state['vectorStore'] is not None:

            result = question_answer()
            response = result({'query': query_area})
            st.write(response)
            st.success("Done...")