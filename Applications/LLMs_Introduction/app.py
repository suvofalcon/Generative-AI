# APPLICATION ON HUGGING FACE SPACES

import streamlit as st
from langchain.llms import OpenAI
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")


def load_answer(question):
    llm = OpenAI(model_name='text-davinci-003', temperature=0)
    answer = llm(question)
    return answer


def get_text():
    input_text = st.text_input("You :", key="input")
    return input_text


user_input = get_text()
response = load_answer(user_input)

submit = st.button("Generate")

# When the submit button is clicked
if submit:
    st.subheader("Answer: ")
    st.write(response)



