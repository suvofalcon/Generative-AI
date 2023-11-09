import streamlit as st
import os
import uuid
from utils import *

# creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = ""


def main():

    st.set_page_config(page_title="Resume Screening Assistance")
    st.title("HR - Resume screening assistance...💁")
    st.subheader("I can help you in resume screening process!!")

    # UI elements
    job_description = st.text_area(
        "Please paste the Job description here.. ", key="1")
    document_count = st.text_input("No of Resumes to return", key="2")

    # Upload the resumes - pdf files only
    pdf = st.file_uploader("Upload resumes here, only PDF files allowed",
                           type=["pdf"], accept_multiple_files=True)

    submit = st.button("Help me with Analysis..")

    # Button action
    if submit:
        with st.spinner("Wait for it..."):

            st.write("our process")
            # creating a unique id , which we can use to query and
            # get only the user uploaded documents from Pinecone vector store
            st.session_state['unique_id'] = uuid.uuid4().hex

            # create a documents list out of all the user uploaded pdf files
            docs = create_docs(pdf, st.session_state['unique_id'])

            # Displaying the count of resumes that have been uploaded
            st.write("Number of Resumes Uploaded - "+str(len(docs)))

            # create embeddings instance
            embeddings = create_embeddings_load_data()

            # push data to pinecone
            push_to_pinecone(os.getenv("PINECONE_API_KEY"),
                             os.getenv("PINECONE_ENVIRONMENT"),
                             "resumes",
                             embeddings, docs)

            # fetch relevant data from pinecone
            relevant_docs = similar_docs(job_description, document_count,
                                         os.getenv("PINECONE_API_KEY"),
                                         os.getenv("PINECONE_ENVIRONMENT"),
                                         "resumes", embeddings, st.session_state['unique_id'])

            st.write(relevant_docs)

            # Introducing a line separator
            st.write(":heavy_minus_sign:" * 30)

            # For each item in relavant docs - we are displaying some info of it on the UI
            for item in range(len(relevant_docs)):

                st.subheader("👉 "+str(item+1))

                # Displaying Filepath
                st.write("**File** : "+relevant_docs[item][0].metadata['name'])

                # Introducing Expander feature
                with st.expander('Show me 👀'):
                    st.info("**Match Score** : "+str(relevant_docs[item][1]))
                    # st.write("***"+relavant_docs[item][0].page_content)

                    # Gets the summary of the current item using 'get_summary'
                    # function that we have created which uses LLM & Langchain chain
                    summary = get_summary(relevant_docs[item][0])
                    st.write("**Summary** : "+summary)

        st.success("Hope I was able to save your time❤️")


# Invoking main function
if __name__ == '__main__':
    main()
