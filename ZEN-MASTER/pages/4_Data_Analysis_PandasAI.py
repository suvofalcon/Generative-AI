# Library imports
#
import streamlit as st
import os
import pandas as pd
from pandasai.llm.openai import OpenAI
# from pandasai import PandasAI
from pandasai import SmartDataframe

st.title("Chat on CSV file - Powered by LLM")


# Function to chat with OpenAI on the CSV through PandasAI
def chat_with_csv(data_frame, prompt):

    # Initialize OpenAI and PandasAI
    llm = OpenAI(api_token=os.getenv("OPENAI_API_KEY"))
    # pandas_ai = PandasAI(llm)
    df = SmartDataframe(data_frame, config={"llm": llm, "verbose": True})
    result = df.chat(prompt)

    # Send the result to LLM
    # result = pandas_ai.run(data_frame, prompt=prompt)
    print(result)
    return result


# Function to use pandasai directly for the data analysis
# def query_on_csv(data_frame, prompt):
#
#    # Initialize OpenAI and PandasAI
#    llm = OpenAI(api_token=os.getenv("OPENAI_API_KEY"))
#    pandasai = PandasAI(llm)
#
#    result = pandasai.run(data_frame, prompt)
#
#    return result


input_csv = st.file_uploader("Upload CSV File", type=[
                             'csv'], accept_multiple_files=False)

if input_csv is not None:

    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("CSV Uploaded Successfully")
        data = pd.read_csv(input_csv)
        st.dataframe(data, use_container_width=True)

    with col2:

        st.info("Ask Question")

        input_query = st.text_area("Enter your Query")
        response_button = st.button("Get Response")

        if input_query is not None:
            if response_button:
                st.info("Your Query: "+input_query)

                # pass the result
                # result = query_on_csv(data, input_query)
                result = chat_with_csv(data, input_query)

                st.write(result)
