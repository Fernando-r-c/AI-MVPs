__author__ = 'fernando'
import json
import os
import sys
import datetime
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

class OpenAIController():
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
        if not all([self.openai_key, self.openai_model]):
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