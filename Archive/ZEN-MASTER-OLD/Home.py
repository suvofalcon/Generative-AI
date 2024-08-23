# Library imports

import streamlit as st

st.set_page_config(page_title="Generative AI powered Applications",
                   page_icon=":robot:", layout='wide')
st.header(
    "Proof of Concepts - Possibilities with Generative AI :robot_face:", divider=True)

st.subheader("Following Technologies are used - ")

st.markdown('''

    - [Streamlit](https://streamlit.io)
    - [LangChain](https://python.langchain.com) Abstraction APIs on LLMs
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model
    - [LLama2 7B Quantized Model](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main) LLM Model
    - [Opensource Embedding (all-MiniLM-L6-V2)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-V2) Sentence Transformer Model from HuggingFace
    - [Vector Store - FAISS](https://ai.meta.com/tools/faiss/) - Facebook AI Similarity Search by Meta

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

Both the above approaches uses FAISS as the vector store for Similarity Search Query

This builds a natural language based interaction using the power of Generative AI on custom documents - such has KEDB, SOPs, Ticket Descriptions and even large text based logs and traces.

Currently the POC implements dealing with multiple PDF files as the knowledge source

 ---

 #### 3 - Data Analysis using LLM

 This uses LLM to put a natural language based query interface on a dataset. This has been currently implemented using CSV files as the data source, but can directly work with database tables and structures - Both relational and NoSQL.

---

#### 4 - Video Script Writer

This uses LLM OpenAI to generate a script and title for any video on a desired object. The content of the script and title is obtained by a thorough search on the given topic using DuckDuckGo Search

---

#### 5 - Work with Knowledge Documents 

This is implented using GooglePalm LLM and GooglePalm embedding using Pinecone as the vector store.
This builds a natural language based interaction using the power of Generative AI on custom documents - such has KEDB, SOPs etc. For now the implementation is dealing with multiple PDF files

''')
