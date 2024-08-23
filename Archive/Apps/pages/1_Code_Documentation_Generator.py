# Library Imports

import streamlit as st
import os
from io import StringIO

from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# Read the OPENAI Key from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Function to read the source file - it will preserve the content format and style
def read_code_file(file):

    if file is not None:
        if file.type == 'text/plain':

            stringio = StringIO(file.getvalue().decode('utf-8'))
            code_data = stringio.read()

    return code_data

# Initialize OpenAI
def initialize_llm():

    llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=1,
                 max_tokens=512, openai_api_key=OPENAI_API_KEY)
    return llm

# Design the code generation prompt
def design_doc_generation_prompt(code_data):

    # language identification
    lang_identify = '''Please identify the programming language given in the code :\n{code_data}

    YOUR RESPONSE:
    '''

    lang_identify_template = PromptTemplate(
        input_variables=['code_data'],
        template=lang_identify
    )

    # Summary Description
    code_summary = '''Please give a brief summary description of the code below :\n{code_data}

    YOUR RESPONSE:
    '''

    code_summary_template = PromptTemplate(
        input_variables=['code_data'],
        template=code_summary
    )

    # Detailed Description
    code_steps = '''Please explain the detailed description of the code point wise and in a step by step
                manner :\n{code_data}

    YOUR RESPONSE
    '''

    code_steps_template = PromptTemplate(
        input_variables=['code_data'],
        template=code_steps
    )

    return lang_identify_template, code_summary_template, code_steps_template

# Construct the required chains and execute the steps

def construct_execute_chain(model, lang_template, summary_template, steps_template,
                            source):

    # construct the chains for sequential execution
    language_chain = LLMChain(llm=model, prompt=lang_template, output_key="lang")
    summary_chain = LLMChain(llm=model, prompt=summary_template, output_key="summary")
    steps_chain = LLMChain(llm=model, prompt=steps_template, output_key="steps")

    # Build the final Chain for execution
    final_chain = SequentialChain(chains=[language_chain, summary_chain, steps_chain],
                                  input_variables=['code_data','code_data','code_data'],
                                  output_variables = ['lang', 'summary', 'steps'],
                                  verbose = True)

    response = final_chain(source)

    return response


# page development
st.title("Code Documentation Generator")
st.subheader("Generate Detailed Documentation for Python, Java, C/C++ and COBOL Files")
st.markdown(''' --- ''')

# File uploader
source_file = st.file_uploader("Upload your Source file(s)",
                               type=['txt'], accept_multiple_files=False)

generate_btn = st.button("Generate Documentation")

if generate_btn:

    # Main processing
    with st.spinner("Getting Details from the source file...."):

        # read the source file contents
        code_contents = read_code_file(source_file)

        # get the prompts
        language, code_summary, code_steps = design_doc_generation_prompt(
            code_data=code_contents
        )

        # initialize the llm
        llm_model = initialize_llm()

        # execute the prompts on llm
        response = construct_execute_chain(model=llm_model, lang_template=language,summary_template=code_summary,
                                           steps_template=code_steps, source=code_contents)

        # Display the output
        st.write(f"The programming language is - {response['lang']}")
        st.markdown(''' --- ''')
        st.write("Summary Description of the Code - \n")
        st.write(response['summary'])
        st.markdown(''' --- ''')
        st.write("Detailed Stepwise Description - \n")
        st.write(response['steps'])
        st.markdown(''' --- ''')

        st.success("Done!!")
