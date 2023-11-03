import streamlit as st
from utils import query_agent

st.title("Lets do some analysis on your CSV")
st.header("Please upload your CSV file here :")

# Capture the CSV file
data = st.file_uploader("Upload csv file", type="csv")

query = st.text_area("Enter your query")
button = st.button("Generate Response..")

if button:
    # Get Response
    answer = query_agent(data, query)
    st.write(answer)