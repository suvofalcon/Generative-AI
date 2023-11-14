from pypdf import PdfReader
import os
import re
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain.llms.openai import OpenAI
import replicate

# Extract information from pdf file


def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract data from text using LLM
def extracted_data(pages_data):

    template = """Extract all the following values : Invoice no., Description, Quantity, Date,

    Unit price, Amount, Total, email, phone number and address from this data - {pages}

    Expected Output - remove any dollar symbols {{'Invoice no.': '1001329','Description': 'Office Chair','Quantity': '2','Date': '5/4/2023','Unit price': '1100.00','Amount': '2200.00','Total': '2200.00','Email': 'Santoshvarma0988@gmail.com','Phone number': '9999999999','Address': 'Mumbai, India'}}
    """
    prompt_template = PromptTemplate(
        input_variables=["pages"], template=template)

    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                 model="text-davinci-003",
                 temperature=0.7)

    full_response = llm(prompt_template.format(pages=pages_data))

    return full_response

# Function to extract data from text using LLM - Llama2


def extracted_data_llama2(pages_data):

    template = """Extract all the following values : Invoice no., Description, Quantity, Date,

    Unit price, Amount, Total, email, phone number and address from this data - {pages}

    Expected Output - remove any dollar symbols {{'Invoice no.': '1001329','Description': 'Office Chair','Quantity': '2','Date': '5/4/2023','Unit price': '1100.00','Amount': '2200.00','Total': '2200.00','Email': 'Santoshvarma0988@gmail.com','Phone number': '9999999999','Address': 'Mumbai, India'}}
    """
    prompt_template = PromptTemplate(
        input_variables=["pages"], template=template)

    output = replicate.run('meta/llama-2-7b-chat:13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0',
                           input={"prompt": prompt_template.format(pages=pages_data),
                                  "temperature": 0.1, "top_p": 0.9,
                                  "max_length": 512, "repetition_penalty": 1})
    full_response = ""
    for item in output:
        full_response += item

    return full_response

# iterate over files in
# that user uploaded PDF files, one by one


def create_docs(user_pdf_list):

    df = pd.DataFrame({'Invoice no.': pd.Series(dtype='str'),
                       'Description': pd.Series(dtype='str'),
                       'Quantity': pd.Series(dtype='str'),
                       'Date': pd.Series(dtype='str'),
                       'Unit price': pd.Series(dtype='str'),
                       'Amount': pd.Series(dtype='int'),
                       'Total': pd.Series(dtype='str'),
                       'Email': pd.Series(dtype='str'),
                       'Phone number': pd.Series(dtype='str'),
                       'Address': pd.Series(dtype='str')
                       })

    for filename in user_pdf_list:

        print(filename)
        raw_data = get_pdf_text(filename)
        # print(raw_data)
        # print("extracted raw data")

        llm_extracted_data = extracted_data(raw_data)
        print("llm extracted data")

        # Adding items to our list - Adding data & its metadata

        pattern = r'{(.+)}'
        match = re.search(pattern, llm_extracted_data, re.DOTALL)

        if match:
            extracted_text = match.group(1)
            # Converting the extracted text to a dictionary
            data_dict = eval('{' + extracted_text + '}')
            print(data_dict)
        else:
            print("No match found.")

        df = df._append([data_dict], ignore_index=True)
        print("********************DONE***************")
        # df=df.append(save_to_dataframe(llm_extracted_data), ignore_index=True)

    df.head()
    return df


# iterate over files one by one - response from llama2

def create_docs_llama2(user_pdf_list):
    llm_extracted_data = []
    for filename in user_pdf_list:
        raw_data = get_pdf_text(filename)

        llm_extracted_data.append(extracted_data_llama2(raw_data))
        print(llm_extracted_data)

    return llm_extracted_data
