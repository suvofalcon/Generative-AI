import streamlit as st
from utils import *
import constants


# Creating Session State variables
if 'HUGGINGFACEHUB_API_TOKEN' not in st.session_state:
    st.session_state['HUGGINGFACEHUB_API_TOKEN'] = ''

if 'PINECONE_API_KEY' not in st.session_state:
    st.session_state['PINECONE_API_KEY'] = ''

# Title of the page
st.title('🤖 AI Assistance For Website')

#********SIDE BAR Funtionality started*******
# Sidebar to capture the API keys
st.sidebar.title("😎🗝️")
st.session_state['HUGGINGFACEHUB_API_TOKEN'] = st.sidebar.text_input("Whats your HuggingFace API Key ?", type="password")
st.session_state['PINECONE_API_KEY'] = st.sidebar.text_input("Whats your Pinecone API Key ?", type="password")

load_button = st.sidebar.button("Load Data to Pinecone", key="load_button")

# If the load_button is clicked, we will push the data to Pinecone

if load_button:
    # Proceed only if API Keys are provided
    if st.session_state['HUGGINGFACEHUB_API_TOKEN'] != "" and st.session_state['PINECONE_API_KEY'] != "":

        # Fetch Data from Site
        site_data = get_website_data(constants.WEBSITE_URL)
        st.write(site_data)
        st.write('Data Pull Complete...')

        #Split Data into chunks
        chunks_data = split_data(site_data)
        st.write('Splitting Data into Chunks - Complete...')

        # Creating Embeddings instance
        embeddings = create_embeddings()
        st.write('Embeddings Instance created... ')

        # Push Data to PineCone
        push_to_pinecone(st.session_state['PINECONE_API_KEY'], constants.PINECONE_ENVIRONMENT,
                         constants.PINECONE_INDEX, embeddings, chunks_data)

        st.sidebar.success('Pushed Data to Pinecone successfully...!!')

    else:
        st.sidebar.error("Please Provide API Keys.... ")

#********SIDE BAR Funtionality ended*******

# Information Retreival

# Captures user's input
prompt = st.text_input('How can I help you my friend ❓', key="prompt") # The box for the text prompt
document_count = st.slider('No.Of links to return 🔗 - (0 LOW || 5 HIGH)', 0, 5, 2,step=1)

submit = st.button("Search")

if submit:
    # Proceed only if API Keys are provided
    if st.session_state['HUGGINGFACEHUB_API_TOKEN'] != "" and st.session_state['PINECONE_API_KEY'] != "":

        # Creating embeddings instance
        embeddings = create_embeddings()
        st.write('Embeddings instance creation done...')

        # Pull index data from Pinecone
        index = pull_from_pinecone(st.session_state['PINECONE_API_KEY'], constants.PINECONE_ENVIRONMENT, constants.PINECONE_INDEX, embeddings)
        st.write("Pinecone Index retieval done...")

        # Fetch relevant documents from pinecone index
        relevant_docs = get_similar_docs(index=index, query=prompt, k=document_count)
        st.write(relevant_docs)

        # Displaying search results
        st.success("Please find the search results :")
        st.write("Search results list...")

        for document in relevant_docs:
            st.write("👉**Result : "+ str(relevant_docs.index(document)+1)+"**")
            st.write("**Info**: "+document.page_content)
            st.write("**Link**: "+ document.metadata['source'])


