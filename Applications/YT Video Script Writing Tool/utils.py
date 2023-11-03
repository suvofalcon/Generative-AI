from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import DuckDuckGoSearchRun


# Function to generate Video Scripts
def generate_script(prompt, video_length, creativity, api_key):
    # Template for generating title
    title_template = PromptTemplate(
        input_variables=['subject'],
        template='Please come up with a title for a YouTube Video on the subject - {subject}'
    )

    # Template for generating 'Video Script' using search engine
    script_template = PromptTemplate(
        input_variables=['title', 'DuckDuckGo_Search', 'duration'],
        template='Create a script for a YouTube Video on this title for me. TITLE - {title} of duration: {duration} minutes using this search data - {DuckDuckGo_Search}'
    )

    # Setting up OpenAI LLM
    llm = OpenAI(temperature=creativity, openai_api_key=api_key, model_name='gpt-3.5-turbo')

    # Creating chain for 'Title' & 'Video Script'
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True)

    # Perform the search for data
    search = DuckDuckGoSearchRun()

    # Executing the chain we created for title
    title = title_chain.run(prompt)

    # Executing the chain we created for script generation
    search_result = search.run(prompt)
    script = script_chain.run(title=title, DuckDuckGo_Search=search_result, duration=video_length)

    # returning the output
    return search_result, title, script
