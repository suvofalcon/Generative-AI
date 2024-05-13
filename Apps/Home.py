# Library Imports

import streamlit as st

# Page Information and configs
st.set_page_config(
    page_title="Generative AI Powered Applications",
    page_icon=":robot:",
    layout="wide"
)

st.header("Simple Applications with Generative AI :robot_face:",
          divider=True)

st.subheader("Following Technologies are used - ")

st.markdown('''

    - [Streamlit](https://streamlit.io)
    - [LangChain](https://python.langchain.com) Abstraction APIs on LLMs
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model

''')

st.markdown('''

### Application Details

#### 1 - Code Documentation Generator

This uses OpenAI Davinci Model (v3.5) to generate Detailed Documentation of Source code. This includes
- Identification of Programming Language.
- Creating a summary of the code functionality
- Detailed Step by Step Explanation of what is happening in each line of the code

This may be useful for teams during Transition to generate documentation for application code, in cases where adequation documentaion is unavailable

This has been currently tested for Python , Java, C/C++ and COBOL only, but may work for other additional programming languages also.

 ---

''')