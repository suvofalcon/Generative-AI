import streamlit as st
import uuid

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
            # creating a unique id , which we can use to query and get only the user uploaded documents from Pinecone vector store
            st.session_state['unique_id'] = uuid.uuid4().hex
            st.write(st.session_state['unique_id'])

            # create a uniqueID, so that we can use to query and get only the user
            # uploaded documents from PINECONE vector store

            # create a documents list out of all the user uploaded pdf files

            # Displaying the count of resumes that have been uploaded

            # create embeddings instance

            # push data to pinecone

            # fetch relevant data from pinecone

        st.success("Hope I was able to save your time❤️")


# Invoking main function
if __name__ == '__main__':
    main()
