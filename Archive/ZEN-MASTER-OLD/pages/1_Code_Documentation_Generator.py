# Library Imports

import streamlit as st
import os
from io import StringIO

from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# Read the OpenAI API Key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Read the source file - as bytes data
def read_source_file(file):

    if file is not None:
        code_data = file.read()

    return code_data


# Read the content of source file - to preserve the code format and style
def read_code_file(file):

    if file is not None:
        if file.type == 'text/plain':

            stringio = StringIO(file.getvalue().decode('utf-8'))
            code_data = stringio.read()

    return code_data


# Initialize OpenAI
def initialize_llm():

    llm = OpenAI(model_name='text-davinci-003', temperature=1,
                 max_tokens=512,
                 openai_api_key=OPENAI_API_KEY)

    return llm

# Design the code generation prompt


def design_doc_generation_prompt(code_data):

    # Prompts design
    lang_identify = """Please identify the programming language given in the code :\n{code_data}

    YOUR RESPONSE:
    """

    lang_identify_template = PromptTemplate(
        input_variables=['code_data'],
        template=lang_identify
    )

    code_summary = """Please give a brief summary description of the code below :\n{code_data}

    YOUR RESPONSE:
    """

    code_summary_template = PromptTemplate(
        input_variables=['code_data'],
        template=code_summary
    )

    code_steps = """Please explain the working of the code below in a detailed point wise step by step manner :\n{code_data}

    YOUR RESPONSE:
    """

    code_steps_template = PromptTemplate(
        input_variables=['code_data'],
        template=code_steps
    )

    return lang_identify_template, code_summary_template, code_steps_template


# Construct the required chains and execute the prompts
def construct_execute_chain(model, lang_template, summary_template,
                            steps_template, source):

    # construct the chains for sequential execution
    language_chain = LLMChain(
        llm=model, prompt=lang_template, output_key="lang")
    summary_chain = LLMChain(
        llm=model, prompt=summary_template, output_key="summary")
    steps_chain = LLMChain(
        llm=model, prompt=steps_template, output_key='steps')

    # Build the final sequential chain
    final_chain = SequentialChain(chains=[language_chain, summary_chain, steps_chain],
                                  input_variables=['code_data',
                                                   'code_data', 'code_data'],
                                  output_variables=[
                                      'lang', 'summary', 'steps'],
                                  verbose=True)

    response = final_chain(source)

    return response


st.title("Code Documentation Generator")
st.subheader(
    "Generate Detailed Documentation for Python, Java, C/C++ and COBOL files")
st.markdown(''' --- ''')
# Build file uploader
source_file = st.file_uploader("Upload your Source File", type=[
                               "txt"], accept_multiple_files=False)

generate_btn = st.button("Generate Documentation")

if generate_btn:

    with st.spinner("Getting Details for the source file ...."):
        # read the source file contents
        code_contents = read_code_file(source_file)

    # get the prompts
        language, summary, detailed_steps = design_doc_generation_prompt(
            code_data=code_contents)

        # initialize the llm
        llm = initialize_llm()

        # execute the prompts on llm
        response = construct_execute_chain(model=llm, lang_template=language,
                                           summary_template=summary, steps_template=detailed_steps,
                                           source=code_contents)

        st.write(f"The programming language is - {response['lang']}")
        st.markdown(''' --- ''')
        st.write("Summary Description of the Code - \n")
        st.write(response['summary'])
        st.markdown(''' --- ''')
        st.write("Detailed Stepwise Description - \n")
        st.write(response['steps'])
        st.markdown(''' --- ''')

        st.success("Done!!")
