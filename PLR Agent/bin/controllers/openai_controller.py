__author__ = 'fernando'

import json
import os
import openai


from langchain import PromptTemplate
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from pydantic import BaseModel, Field

class DocumentInput(BaseModel):
    question: str = Field()

class OpenAIController(object):
    """
    OpenAI Controller class to interact with the OpenAI API.
    Inspired and addapted from https://docs.llamaindex.ai/en/stable/examples/agent/multi_document_agents/
    and https://python.langchain.com/v0.1/docs/integrations/toolkits/document_comparison_toolkit/
    """

    def __init__(self, temperature=0):
        """
        Constructor for the OpenAI Controller class.
        :param temperature: temperature for the OpenAI API
        """
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.openai_model = "gpt-3.5-turbo"
        self.openai_gpt4_model = "gpt-4o"
        self.openai_embedding_model = "text-embedding-ada-002"
        self.__validate_openai_config()
        
        #Configure the OpenAI API
        openai.api_key = self.openai_key
        self.llm_chat = self.__get_chat_language_model(temperature, self.openai_gpt4_model)
        self.chat_embedding_model = self.__get_chat_embedding_model()
        #Load the vector stores

    def __validate_openai_config(self):
        """
        Validates the OpenAI configuration.
        """
        if not all([self.openai_key, self.openai_gpt4_model]):
            raise ValueError("OpenAI key is missing in the configuration file.")
    
    def __get_chat_language_model(self, temperature, model):
        """
        Gets the OpenAI chat language model.
        :param temperature: temperature for the OpenAI API
        :param model: model for the OpenAI API
        :return: OpenAI chat language model
        """
        print(f"Getting OpenAI chat model: {model}")
        return ChatOpenAI(
            openai_api_key=self.openai_key,
            model=model,
            temperature=temperature,
        )
    
    def __get_chat_embedding_model(self):
        """
        Gets the OpenAI chat embedding model.
        :return: OpenAI chat embedding model
        """
        print(f"Getting OpenAI chat embedding model: {self.openai_embedding_model}")
        return OpenAIEmbeddings(
            model=self.openai_embedding_model,
            openai_api_key=self.openai_key
        )
    
    def extract_keywords(self, query, youtube_controller, max_results=5):
        """
        Extracts keywords from a query.
        :param query: query
        :param youtube_controller: YoutubeController
        :param max_results: maximum number of results
        :return: extracted keywords
        """
        print(f"Extracting keywords from query: {query}")
        response_list = []
        try:
            tools = [
                Tool.from_function(
                    func=lambda query: youtube_controller.search_educational_videos(query, max_results),
                    name="youtube_search",
                    description=f"""
                    Search for educational videos on Youtube. 
                    Give me a list of {max_results} video ids in a list.
                    """,
                ),
            ]
            agent = initialize_agent(
                agent=AgentType.OPENAI_FUNCTIONS,
                llm=self.llm_chat,
                tools=tools,
                verbose=True,
                #TODO: use json mode from gpt 4o
            )
            print("Agent initialized for keyword extraction.")
            if agent:
                prompt = self.__get_plr_agent_prompt(query, max_results)
                response_str = agent.run(input=prompt, chat_history=None)
                print(f"Response string: {response_str}")
                if "```json" in response_str:
                    response_str = response_str.split("```json")[1].split("```")[0].strip()
                response_list = json.loads(response_str.strip())
                print(f"Response list: {response_list}")
        except Exception as e:
            print(f"Error initializing agent: {e}")
            raise Exception(f"Error initializing agent: {e}")
        return response_list.get('response', [])
    
    def __get_plr_agent_prompt(self, query, max_results):
        """
        Gets the PLR Agent prompt.
        :param query: query
        :param max_results: maximum number of results
        :return: prompt
        """
        prompt = """Recommend {max_results} educational videos on Youtube about the keywords in user's input =
                 ' {query} '
            Use the provided tools and return a json response list from the Youtube search tool in the format:
            {{{{"response":
                [
                    {{{{
                        "title": "Video Title",
                        "description": "Video Description",
                        "link": "Video Link"
                    }}}},
                    {{{{
                        "title": "Video Title",
                        "description": "Video Description",
                        "link": "Video Link"
                    }}}}, ...
                ]
            }}}}
            """
        return prompt.format(max_results=max_results, query=query)