# Imports
#
import streamlit as st
import os
from PyPDF2 import PdfReader
from model import create_chunks_from_pdfText, create_vector_store_from_chunks, perform_search_return_results

openai_key = os.getenv("OPENAI_API_KEY")

# Function to get pdftext from all the uploaded PDF Files
def get_pdf_text(pdf_docs):

    text = ""
    for pdf in pdf_docs: # read every pdf in the list uploaded
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages: # for every page in the pdf being read
            text += page.extract_text()

    return text


# The UI begins here

def main():

    st.header("Chat with Multiple PDF Files :books:")

    text_input = st.text_input("Enter your Query")
    response_btn = st.button("Generate Response")
    
    if "conversation" not in st.session_state:
        st.session_state['conversation'] = ''

    if "vectorStore" not in st.session_state:
        st.session_state['vectorStore'] = ''

    with st.sidebar:
        st.header("Chat with PDF")
        st.header("LLM Chatapp using LangChain")
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
