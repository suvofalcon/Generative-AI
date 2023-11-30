# Library imports

import streamlit as st
import os

from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import DuckDuckGoSearchRun

# Function to generate Video Scripts
def generate_video_script(prompt, video_length, creativity, api_key):
    
    # Template for generating title
    title_template = PromptTemplate(
        input_variables=['subject'],
        template = 'Please come up with a title for a Video on the subject - {subject}'
    )

    # Template for generating video script using search engine
    script_template = PromptTemplate(
        input_variables = ['title', 'DuckDuckGo_search', 'duration'],
        template = 'Create a script for a Video on this title for me. Title - {title}, of Duration - {duration} minutes, using this search data - {DuckDuckGo_search}'
    )

    # Setting up the OpenAI LLM
    llm = OpenAI(openai_api_key = api_key, temperature=creativity)
    
    # Creating a chain for Title and Video script
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True)
    
    # Perform the search for data
    search = DuckDuckGoSearchRun()

    # Executing the chain we created for title
    title = title_chain.run(prompt)

    # Executing the chain we created for script generation
    search_result = search.run(prompt)
    script = script_chain.run(title=title, DuckDuckGo_search=search_result, duration=video_length)

    # returning the output
    return search_result, title, script



# Applying Styling
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #00ff00;
    color:#FFFFFF;
    }
</style>""", unsafe_allow_html=True)

st.title("Video Script Generator :cinema:")

# Captures the user's input
prompt = st.text_input('Please provide the topic of the video', key="prompt")  # The box for the text prompt
video_length = st.text_input('Expected Video Length 🕒 (in minutes)', key="video_length")  # The box for the text prompt
creativity = st.slider('Words limit ✨ - (0 LOW || 1 HIGH)', 0.0, 1.0, 0.2, step=0.1)

submit = st.button("Generate Script for me")

if submit:

    # Lets generate the script
    search_result, title, script = generate_video_script(prompt=prompt, video_length=video_length, creativity=creativity, 
                                                         api_key = os.getenv("OPENAI_API_KEY"))

    # Display Title
    st.subheader("Title:🔥")
    st.write(title)

    # Display Video Script
    st.subheader("Your Video Script:📝")
    st.write(script)

    # Display Search Engine result
    st.subheader("Check Out - DuckDuckGo Search:🔍")
    with st.expander('Show me 👀'):
        st.info(search_result)
 

