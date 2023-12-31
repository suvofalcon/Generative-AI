import pandas as pd
from langchain.llms import OpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

def query_agent(data, query):
    # Parse the CSV file and create a pandas dataframe from its contents
    df = pd.read_csv(data)

    # Initialize the llm
    llm = OpenAI()

    # create a pandas dataframe agent
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    '''
    Python REPL: A Python shell used to evaluating and executing Python commands.
    It takes python code as input and outputs the result. The input python code can be generated from another tool in the LangChain
    '''
    return agent.run(query)  # returning the response that is coming for our query

