import os
from dotenv import load_dotenv
from pathlib import Path
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain import chat_models
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_party.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

# Retrieve the open api key from OpenAI
load_dotenv(Path(".env"))
openai_api_key = os.getenv("OPEN_API_KEY")

if __name__ == "__main__":
    print("Hello Langchain")

    linkedin_profile_url = linkedin_lookup_agent(name="Eden Marco")

    summary_template = """
        give the linkedin information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(
        temperature=0.9, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key
    )

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
    print(chain.run(information=linkedin_data))
