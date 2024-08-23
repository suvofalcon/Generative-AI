# Library Imports #############################################################

import streamlit as st
import os
import pandas as pd

from langchain.llms.openai import OpenAI
import pandasai
from pandasai import SmartDataframe
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Some Environment Constants ##################################################
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Utility Functions ###########################################################

# Function to query Pandas agent directly
def query_agent(data, query):
    # Parse the CSV file and create a pandas dataframe from its contents
    df = pd.read_csv(data)

    # Initialize the llm
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, max_tokens=256)

    # create a pandas dataframe agent
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    # A Python shell used to evaluating and executing Python commands.
    # It takes python code as input and outputs the result. The input python
    # code can be generated from another tool in the LangChain
    # returning the response that is coming for our query
    return agent.run(query)


# Function to chat with OpenAI on the CSV through PandasAI
def chat_with_csv(data_frame, prompt):

    # Initialize OpenAI and PandasAI
    llm = pandasai.llm.OpenAI(api_token=os.getenv("OPENAI_API_KEY"))
    # pandas_ai = PandasAI(llm)
    df = SmartDataframe(data_frame, config={"llm": llm, "verbose": True})
    result = df.chat(prompt)

    # Send the result to LLM
    # result = pandas_ai.run(data_frame, prompt=prompt)
    print(result)
    return result


# UI Begins from here..########################################################
st.title("Chat on CSV File :chart: - Powered by LLM")
st.subheader("Please Upload your CSV File...:file_folder: ")

input_csv = st.file_uploader("Upload CSV File", type=[
    'csv'], accept_multiple_files=False)

if input_csv is not None:

    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("CSV Uploaded Successfully")
        data = pd.read_csv(input_csv)
        st.dataframe(data, use_container_width=True)

    with col2:

        st.info("Query to LLM")

        input_query = st.text_area("Enter your Query")

        col3, col4 = st.columns([1, 6])

        with col3:

            response_button_openai = st.button(
                "Get Response - OpenAI", key="response_button_openai")

        with col4:

            response_button_pandasai = st.button("Get Response - PandasAI",
                                                 key="response_button_pandasai")

        if response_button_openai:
            if input_query is not None:

                with st.spinner("Getting response from OpenAI..."):
                    input_csv.seek(0)
                    result = query_agent(input_csv, input_query)

                    st.write(result)

                    st.success("Done...")

        if response_button_pandasai:
            if input_query is not None:

                with st.spinner("Getting Response from PandasAI..."):

                    # Getting response from pandasai
                    input_csv.seek(0)
                    data = pd.read_csv(input_csv)
                    result = chat_with_csv(data, input_query)
                    st.write(result)

                    st.success("Done...")
