import streamlit as st
from pages.admin_utils import *


def main():
    st.set_page_config(page_title="Dump PDF to Pinecone - Vectore Store")
    st.title("Please upload your files...📁 ")

    # Upload the pdf file
    pdf = st.file_uploader("Only PDF Files allowed", type=["pdf"])

    # Extract the whole text from the uploaded pdf file
    if pdf is not None:
        with st.spinner("Wait for it..."):

            text = read_pdf_data(pdf_file=pdf)
            st.write("👉Reading PDF Done ...")

            # Create Chunks
            docs_chunks = split_data(text=text)
            st.write("👉Splitting data into chunks done...")

            # Create the embeddings
            embeddings = create_embeddings_load_data()
            st.write("👉Creating embedding instance done...")

            # Build the Vector Store and Push the PDF data embeddings
            push_to_pinecone(os.getenv('PINECONE_API_KEY'), os.getenv('PINECONE_ENVIRONMENT'),
                             'tickets', embeddings, docs_chunks)

        st.success("Successfully pushed the embeddings to Pinecone")

if __name__ == '__main__':
    main()

