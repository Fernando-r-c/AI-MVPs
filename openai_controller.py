__author__ = 'fernando'
import json
import os
import sys
import datetime
import openai
sys.path.insert(0, "bin/common/")
from ini_config_reader import read_ini_config_section, DEFAULT_CONFIG_FILE
sys.path.insert(0, 'bin/common/utils/')
from logging_util import LoggingUtil
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
    VECTOR_STORE_PATH = "resources/document_comparison/vector_stores/"
    EXTERNAL_DOCUMENTS_PATH = "resources/document_comparison/T&C/external/"
    TMP_EXTERNAL_DOCUMENTS_PATH = "tmp/document_comparison/T&C/external/"
    def __init__(self, config_file=DEFAULT_CONFIG_FILE, section='openai', temperature=0):
        """
        Constructor for the OpenAI Controller class.
        :param config_file: configuration file
        :param section: section of the configuration file
        :param temperature: temperature for the OpenAI API
        """
        self.openai_config = read_ini_config_section(filename=config_file, section=section)
        self.openai_key = self.openai_config.get('openai_key')
        self.openai_model = self.openai_config.get('openai_3_5_turbo_model')
        self.openai_gpt4_model = self.openai_config.get('openai_gpt4_model')
        self.openai_embedding_model = self.openai_config.get('openai_embedding_model')
        self.__validate_openai_config()
        if not os.path.exists(self.VECTOR_STORE_PATH):
            os.makedirs(self.VECTOR_STORE_PATH)
        if not os.path.exists(self.EXTERNAL_DOCUMENTS_PATH):
            os.makedirs(self.EXTERNAL_DOCUMENTS_PATH)
        
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
        LoggingUtil.display_text(f"Getting OpenAI chat model: {model}")
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
        LoggingUtil.display_text(f"Getting OpenAI chat embedding model: {self.openai_embedding_model}")
        return OpenAIEmbeddings(
            model=self.openai_embedding_model,
            openai_api_key=self.openai_key
        )
    
    def save_external_document(self, document_file, tmp=False):
        """
        Save an external document into the resources/T&C/external folder.
        :param document_file: document file
        """
        LoggingUtil.display_text(f"Downloading external document: {document_file}")
        try:
            filename = document_file.filename if hasattr(document_file, 'filename') else document_file
            absolute_document_path = self.__get_absolute_path(filename, tmp)
            document_file.save(absolute_document_path) if hasattr(document_file, 'save') else open(absolute_document_path, 'wb').write(document_file)
            LoggingUtil.display_text(f"Saved external document to: {absolute_document_path}")
            return absolute_document_path
        except Exception as e:
            LoggingUtil.display_text(f"Error saving external document: {e}")
            raise Exception(f"Error saving external document")
        
    def __get_absolute_path(self, filename, tmp=False):
        """
        Gets the absolute path for a filename.
        :param filename: filename
        :return: absolute path
        """
        path = self.TMP_EXTERNAL_DOCUMENTS_PATH if tmp else self.EXTERNAL_DOCUMENTS_PATH
        absolute_path = os.path.join(path, filename)
        if os.path.exists(absolute_path):
            os.remove(absolute_path)
        return absolute_path
    
    def create_and_save_vector_store(self, vector_store_name, document):
        """
        Creates and saves a vector store for a document.
        :param vector_store_name: name of the vector store
        :param document: document
        """
        LoggingUtil.display_text(f"Creating and saving vector store for document: {document}")
        start_time = datetime.datetime.now()
        vector_store_status = False
        try:
            if vector_store_name.endswith('pdf'):
                loader = PyPDFLoader(document)
            elif vector_store_name.endswith('txt'):
                TextLoader(document)
                loader = TextLoader(document)
            elif vector_store_name.endswith('<html>'):
                WebBaseLoader(document)
                loader = WebBaseLoader(document)
            pages = loader.load_and_split()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(pages)
            embeddings = self.chat_embedding_model
            retriever = FAISS.from_documents(docs, embeddings)
            retriever.save_local(self.VECTOR_STORE_PATH + vector_store_name)
            if os.path.exists(self.VECTOR_STORE_PATH + vector_store_name):
                vector_store_status = True
        except Exception as e:
            LoggingUtil.display_text(f"Error creating and saving vector store: {e}")
            raise Exception(f"Error creating and saving vector store")
        finally:
            end_time = datetime.datetime.now()
            LoggingUtil.display_elapsed_time(start_time, end_time, 'create_and_save_pdf_vector_store', flush_output=True) 
            return vector_store_status
        
    def __load_vector_store_as_retriever(self, vector_store_name):
        """
        Loads a vector store as a retriever.
        :param vector_store_name: name of the vector store
        :return: retriever
        """
        LoggingUtil.display_text(f"Loading vector store as retriever: {vector_store_name}")
        try:
            vector_store = FAISS.load_local(self.VECTOR_STORE_PATH + vector_store_name, embeddings=self.chat_embedding_model)
            LoggingUtil.display_text(f"Vector store loaded as retriever: {vector_store_name}")
            return vector_store.as_retriever()
        except Exception as e:
            LoggingUtil.display_text(f"Error loading vector store as retriever: {e}")
            raise Exception(f"Error loading vector store as retriever: {e}")
        
    def __get_tools(self, document_1_name, document_2_name):
        """
        Gets the tools.
        """
        tools = []
        try:
            for file in [document_1_name, document_2_name]:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm_chat, 
                    retriever=self.__load_vector_store_as_retriever(file),
                )
                tools.append(
                    Tool(
                        args_schema=DocumentInput,
                        name=f"{file}",
                        description=f"useful when you want to answer questions about {file}",
                        func=qa_chain.run,
                    )
                )
            LoggingUtil.display_text(f"Tools: {tools}")
        except Exception as e:
            LoggingUtil.display_text(f"Error getting tools: {e}")
            raise Exception(f"Error getting tools: {e}")
        return tools
    
    def __get_tools_text(self):
        """
        Gets the tools without requiring specific document names.
        """
        tools = []
        try:
            # Create a single QA chain tool for general text input
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm_chat,
                retriever=self.__create_general_retriever()  # Create a new method for general retriever
            )
            
            tools.append(
                Tool(
                    args_schema=DocumentInput,  # Ensure this schema is appropriate for general text input
                    name="text_input_comparison",
                    description="Useful when you want to answer questions about provided text inputs.",
                    func=qa_chain.run,
                )
            )
            
            LoggingUtil.display_text(f"Tools: {tools}")
        except Exception as e:
            LoggingUtil.display_text(f"Error getting tools: {e}")
            raise Exception(f"Error getting tools: {e}")
    
        return tools
    
    def __initialize_agent_text(self):
        """
        Initializes the agent without requiring input document names.
        """
        try:
            # Get the necessary tools for the agent without needing document names
            tools = self.__get_tools_text()  # Update this method if needed to remove parameters
            agent = initialize_agent(
                agent=AgentType.OPENAI_MULTI_FUNCTIONS,
                tools=tools,
                llm=self.llm_chat,
                verbose=True,
            )
            LoggingUtil.display_text("Agent initialized without document inputs.")
        except Exception as e:
            LoggingUtil.display_text(f"Error initializing agent: {e}")
            raise Exception(f"Error initializing agent: {e}")
        return agent
    
    def __initialize_agent(self, document_1_name, document_2_name):
        """
        Initializes the agent.
        """
        try:
            tools = self.__get_tools(document_1_name, document_2_name)
            agent = initialize_agent(
                agent=AgentType.OPENAI_MULTI_FUNCTIONS,
                tools=tools,
                llm=self.llm_chat,
                verbose=True,
            )
            LoggingUtil.display_text(f"Agent initialized in function for {document_1_name} and {document_2_name}")
        except Exception as e:
            LoggingUtil.display_text(f"Error initializing agent: {e}")
            raise Exception(f"Error initializing agent: {e}")
        return agent
    
    def __get_prompt_template(self, base_question):
        """
        Gets the prompt template for the agent.
        :param base_question: base question for the agent
        """
        return (
                "You are an expert document comparison assistant. "
                "Please provide your response in the following JSON format:\n\n"
                "{{"
                "\"key_differences\": [\n"
                "    {{\n"
                "        \"difference\": \"\",\n"
                "       \"source\": {{\n"
                "        \"document\": \"\",\n"
                "        \"section\": \"\",\n"
                "        \"topic\": \"\"\n"
                "      }}\n"
                "    }}\n"
                "],\n"
                "\"benefits_drawbacks\": {{\n"
                "    \"benefits\": \"\",\n"
                "    \"drawbacks\": \"\"\n"
                "}},\n"
                "\"executive_summary\": \"\"\n"
                "}}\n\n"
                "Use the provided tools to fill in the JSON fields appropriately.\n\n"
                "The source field should contain the document name along with the section or topic name.\n\n"
                "If you don't have any information to provide, leave the JSON fields empty.\n\n"
                f"Question: {base_question}\n\n"
            )
    
    def query_agent_file(self, document_1_name, document_2_name, query=None):
        """
        Queries the agent.
        :param query: query to be executed
        :return: results
        """
        LoggingUtil.display_text(f"Querying the agent for {document_1_name} and {document_2_name}")
        start_time = datetime.datetime.now()
        try:
            # Handle both PDF and TXT file types
            for doc_name in [document_1_name, document_2_name]:
                if doc_name.endswith('pdf'):
                    self.create_and_save_vector_store(doc_name, self.EXTERNAL_DOCUMENTS_PATH + doc_name)
                elif doc_name.endswith('txt'):
                    self.create_and_save_vector_store(doc_name, self.EXTERNAL_DOCUMENTS_PATH + doc_name)
                else:
                    raise ValueError(f"Unsupported file type for {doc_name}. Only PDF and TXT files are supported.")
            # Initialize OpenAI callback and agent for document comparison
            with get_openai_callback() as callback:
                agent = self.__initialize_agent(document_1_name, document_2_name)
                LoggingUtil.display_text(f"Agent initialized for {document_1_name} and {document_2_name}")
                
                # Base system instruction for the agent
                base_question = f"What are the key points of {document_1_name} and {document_2_name}?"
                input_str = self.__get_prompt_template(base_question)
                LoggingUtil.display_text(f"Prompt template: {input_str}")
                
                if query:
                    input_str += f" Additionally, {query}"
                
                # Run the agent with the constructed input
                agent_response = agent.run(input_str)
                # Extract token usage and cost from callback
                total_tokens = callback.total_tokens
                prompt_tokens = callback.prompt_tokens
                completion_tokens = callback.completion_tokens
                total_cost = callback.total_cost
                LoggingUtil.display_text(f"Total tokens: {total_tokens}, Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total cost: {total_cost}")
                # Parse JSON from the agent response, if applicable
                if "```json" in agent_response:
                    agent_response = json.loads(agent_response.split("```json")[1].split("```")[0].strip())
                LoggingUtil.display_text(f"Agent Response: {agent_response}")
                
        except Exception as e:
            LoggingUtil.display_text(f"Error querying agent: {e}")
            raise Exception(f"Error querying agent: {e}")
        except json.JSONDecodeError:
            LoggingUtil.display_text("Failed to parse agent response as JSON.")
            raise Exception("Agent response is not in valid JSON format.")
        
        end_time = datetime.datetime.now()
        LoggingUtil.display_elapsed_time(start_time, end_time, 'query_agent', flush_output=True)
        return agent_response
    
    def query_agent(self,document_1_name, document_2_name, query=None):
        """
        Queries the agent.
        :param query: query to be executed
        :return: results
        """
        LoggingUtil.display_text(f"Querying the agent for {document_1_name} and {document_2_name}")
        start_time = datetime.datetime.now()
        try:
            with get_openai_callback() as callback:
                agent = self.__initialize_agent(document_1_name, document_2_name)
                # Base system instruction for the agent
                base_question = f"What are the key points of {document_1_name} and {document_2_name}?"
                input_str = self.__get_prompt_template(base_question)
                if query:
                    input_str += f" Additionally, {query}"
                # Run the agent with the constructed input
                agent_response = agent.run(input_str)
                total_tokens = callback.total_tokens
                prompt_tokens = callback.prompt_tokens
                completion_tokens = callback.completion_tokens
                total_cost = callback.total_cost
                LoggingUtil.display_text(f"Total tokens: {total_tokens}, Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total cost: {total_cost}")
                if "```json" in agent_response:
                    agent_response = json.loads(agent_response.split("```json")[1].split("```")[0].strip())
                LoggingUtil.display_text(f"Agent Response: {agent_response}")
        except Exception as e:
            LoggingUtil.display_text(f"Error querying agent: {e}")
            raise Exception(f"Error querying agent: {e}")
        except json.JSONDecodeError:
            LoggingUtil.display_text("Failed to parse agent response as JSON.")
            raise Exception("Agent response is not in valid JSON format.")
        end_time = datetime.datetime.now()
        LoggingUtil.display_elapsed_time(start_time, end_time, 'query_agent', flush_output=True)
        return agent_response
    
    def extract_keywords(self, query, youtube_controller, max_results=5):
        """
        Extracts keywords from a query.
        :param query: query
        :param youtube_controller: YoutubeController
        :param max_results: maximum number of results
        :return: extracted keywords
        """
        LoggingUtil.display_text(f"Extracting keywords from query: {query}")
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
            )
            LoggingUtil.display_text("Agent initialized for keyword extraction.")
            if agent:
                prompt = self.__get_plr_agent_prompt(query, max_results)
                response_str = agent.run(input=prompt, chat_history=None)
                LoggingUtil.display_text(f"Response string: {response_str}")
                if "```json" in response_str:
                    response_str = response_str.split("```json")[1].split("```")[0].strip()
                response_list = json.loads(response_str.strip())
                LoggingUtil.display_text(f"Response list: {response_list}")
        except Exception as e:
            LoggingUtil.display_text(f"Error initializing agent: {e}")
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
    
if __name__ == '__main__':
    #test some queries on our base index
    LoggingUtil.display_text("Starting OpenAI Controller...")
    start_time = datetime.datetime.now()
    openai_controller = OpenAIController()
    #! Doc NAMES must be unique and with only alphanumeric characters: '^[a-zA-Z0-9_-]+$'
    #Google T&C documents
    GOOGLE_1_PATH = "resources/document_comparison/T&C/Google/google_terms_of_service_en_us_2022_Jan_05_age.pdf"
    GOOGLE_2_PATH = "resources/document_comparison/T&C/Google/google_terms_of_service_en_us_2024_May_22_expect.pdf"
    GOOGLE_1_NAME = "google-tc-2022-Jan-05-age"
    GOOGLE_2_NAME = "google-tc-2024-May-22-expect"
    #OpenAI T&C documents
    OPENAI_1_PATH = "resources/document_comparison/T&C/OpenAI/openai_terms_of_service_en_us_2023_Mar_14.pdf"
    OPENAI_2_PATH = "resources/document_comparison/T&C/OpenAI/openai_terms_of_service_en_us_2023_Nov_14.pdf"
    OPENAI_1_NAME = "openai-tc-2023-Mar-14"
    OPENAI_2_NAME = "openai-tc-2023-Nov-14"
    #AWS T&C documents
    AWS_1_PATH = "resources/T&C/AWS/aws_terms_of_service_en_us_2023_Jul_06.pdf"
    AWS_2_PATH = "resources/T&C/AWS/aws_terms_of_service_en_us_2024_May_17.pdf"
    AWS_1_NAME = "aws-tc-2023-Jul-06"
    AWS_2_NAME = "aws-tc-2024-May-17"
    #create and save the vector stores
    #LoggingUtil.display_text("Creating and saving vector stores...")
    openai_controller.create_and_save_vector_store(GOOGLE_1_NAME, GOOGLE_1_PATH)
    openai_controller.create_and_save_vector_store(GOOGLE_2_NAME, GOOGLE_2_PATH)
    #queries to test
    query = ""
    #langchain_results = openai_controller.query_agent(GOOGLE_1_NAME, GOOGLE_2_NAME, query)
    #query the base engine for llama index approach
    #results = openai_controller.query_base_engine(query)
    #display the results
    #LoggingUtil.display_text(f"Results:\n {results}")
    end_time = datetime.datetime.now()
    LoggingUtil.display_elapsed_time(start_time, end_time, 'Testing OpenAI Controller', flush_output=True)