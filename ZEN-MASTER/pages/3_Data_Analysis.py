# Library Imports

import streamlit as st
import os
import pandas as pd

from langchain.llms.openai import OpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


# Get the Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def query_agent(data, query):
    # Parse the CSV file and create a pandas dataframe from its contents
    df = pd.read_csv(data)

    # Initialize the llm
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)

    # create a pandas dataframe agent
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    # A Python shell used to evaluating and executing Python commands.
    # It takes python code as input and outputs the result. The input python
    # code can be generated from another tool in the LangChain
    # returning the response that is coming for our query
    return agent.run(query)


# UI Begins from here...

st.title("Data Analysis on CSV 🔍")
st.subheader("Please Upload your CSV File...:file_folder: ")

# Capture the CSV file
data = st.file_uploader("Upload CSV File", type=[
                        'csv'], accept_multiple_files=False)

query = st.text_area("Enter your Query")
button = st.button("Response")

if button:
    if query is not None:
        # Get Response
        answer = query_agent(data, query)
        st.write(answer)
