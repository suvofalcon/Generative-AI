# Library imports

import streamlit as st
import os

st.set_page_config(page_title="Generative AI powered Applications", page_icon=":robot:", layout='wide')
st.header("Proof of Concepts - Possibilities with Generative AI", divider=True)

st.subheader("Following Technologies are used - ")

st.markdown('''

    - [Streamlit](https://streamlit.io)
    - [LangChain](https://python.langchain.com) Abstraction APIs on LLMs
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model
    - [LLama2 7B Quantized Model](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main) LLM Model
    - [Opensource Embedding (all-MiniLM-L6-V2)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-V2) Sentence Transformer Model from HuggingFace
    - [Vector Store - FAISS](https://ai.meta.com/tools/faiss/) - Facebook AI Similarity Search

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

#### 2 - Work with Knowledge Documents

This has two implementations
- Using OpenAI Embedding and OpenAI Davinci Model   
- Using open source embedding Model from HuggingFace and LLama2 7B Quantized Model
- This also uses FAISS as the vector store for Similarity Search Query

This builds a natural language based interation using the power of Generative AI on custom documents , such has KEDB, SOPs, Ticket Descriptions and even large text based logs and traces.

 ---

 #### 3 - Data Analysis using LLM

 This uses LLM to put a natural language based query interface on a dataset. This has been currently implemented using CSV files as the data source, but can directly work with database tables and structures - Both relational and NoSQL.

---
''')
