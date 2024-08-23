# Library imports
import os
from langchain import embeddings
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import retriever
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl


# Get the vectorstores db path
DB_FAISS_PATH = os.path.join(os.getenv("HOME"),
                             "vectorstores/medicalbot/db_faiss")

MODEL_PATH = os.path.join(
    os.getenv("HOME"), "models/llama-2-7b-chat.ggmlv3.q8_0.bin")

# define the custom prompt template
custom_prompt_template = """Use the following pieces of information to answer user's question.
In case you dont know the answer, just say you dont know the answer, dont try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.

Helpful answer:
"""

# function to set custom prompt


def set_custom_prompt():

    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])

    return prompt

# function to load the llm


def load_llm():
    # load the locally downloaded model here
    llm = CTransformers(model=MODEL_PATH,
                        model_type="llama",
                        max_new_tokens=2048,
                        temperature=0.5)
    return llm

# Retrieval QA Chain


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(
                                               search_kwargs={'k': 2}),
                                           return_source_documents=True,  # to explain the output to the end user
                                           chain_type_kwargs={'prompt': prompt})
    return qa_chain


# question - answer
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # now we have to load from faiss
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    # load the llm
    llm = load_llm()

    # set the custom prompt
    qa_prompt = set_custom_prompt()

    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# output function


def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


# Chainlit functions
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content='Starting the bot...')
    await msg.send()
    msg.content = "Hi, Welcome to the medical bot. What is your query ?"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()
