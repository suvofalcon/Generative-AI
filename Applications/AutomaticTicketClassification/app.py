import os
import streamlit as st
from user_utils import *

#Creating session variables
if 'HR_tickets' not in st.session_state:
    st.session_state['HR_tickets'] =[]
if 'IT_tickets' not in st.session_state:
    st.session_state['IT_tickets'] =[]
if 'Transport_tickets' not in st.session_state:
    st.session_state['Transport_tickets'] =[]

def main():

    st.header("Automatic Ticket Classification Tool")
    # Captures user's input
    st.write("We are here to help you, please ask your question:")
    user_input = st.text_input("🔍")

    if user_input:

        # creating embeddings instance
        embeddings = create_embeddings()

        # Function to pull index data from Pinecone
        index = pull_from_pinecone(pinecone_api_key=os.getenv("PINECONE_API_KEY"), pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
                                   pinecone_index_name='tickets',embeddings=embeddings)

        # This function will help us in fetching the top relevant documents from our vector store - pinecone index
        relevant_docs = get_similar_docs(index=index,query=user_input,k=2)

        # This will return the fine tuned response by LLM
        response = get_answer(docs=relevant_docs,query=user_input)
        st.write(response)

        # Button to create a ticket with respective department
        button = st.button("Submit Ticket ?")

        if button:
            # Get Response

            embeddings = create_embeddings()
            query_result = embeddings.embed_query(user_input)

            # loading the ML model, so that we can use it to predict the class where this complaint can belong to,,
            department_value = predict(query_result=query_result)
            st.write(f"The ticket has been submitted to : {department_value}")

             #Appending the tickets to below list, so that we can view/use them later on...
            if department_value=="HR":
                st.session_state['HR_tickets'].append(user_input)
            elif department_value=="IT":
                st.session_state['IT_tickets'].append(user_input)
            else:
                st.session_state['Transport_tickets'].append(user_input)


if __name__ == '__main__':
    main()