'''
This is the streamlit application
'''

# Library imports
import os
import streamlit as st
from model import create_chunks_from_pdf, create_embeddings_from_text, perform_search_return_results

# We will build the UI

def main():
    st.header('Chat with  PDF 💬')
    st.sidebar.title('LLM Chat App with LangChain')

    st.sidebar.markdown('''
    This is a LLM Powered chatbot built using:
    - [Streamlit](https://streamlit.io)
    - [LangChain](https://python.langchain.com)
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model
    ''')

    # File uploader
    pdf = st.file_uploader("Upload your PDF File", type='pdf')

    # Query area
    query = st.text_input("Ask Question from your PDF File!")
    button = st.button("Response")

    if button:
        
        # Get the api key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # call for creating chunks
        chunks = create_chunks_from_pdf(pdf_file=pdf)

        # create embeddings
        data_store = create_embeddings_from_text(openai_api_key, chunks)

        # We will the search on the store and get the response from LLM
        response = perform_search_return_results(key=openai_api_key, query=query, vector_store=data_store)

        st.write(response)


if __name__ == "__main__":
    main()
