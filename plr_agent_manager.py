__author__ = 'fernando'

import json
import os
import sys
import datetime

from io import BytesIO

sys.path.insert(0, 'bin/common/utils/')
from logging_util import LoggingUtil

sys.path.insert(0, 'bin/controllers/')
from openai_controller import OpenAIController
from youtube_controller import YoutubeController

# all api versions listed below MUST be in lowercase
VERSION_1 = 'v1'  
'''
all versions listed in the set MUST be defined in:
    - query_plr_agent()
'''

VALID_API_VERSION_SET = {VERSION_1}

class PLRAgentManager(object):
    """
    Class to manage the Personalized Learning Recommendations Agent
    """

    RESPONSE_KEY = 'response'
    SUCCESS_KEY = 'success'
    RECOMMENDATIONS_KEY = 'recommendations'
    ERROR = 'error'
    
    def __init__(self, api_version):
        """
        Initialize the PLR Agent Manager object.
        :param api_version: api version to use
        """
        start_time = datetime.datetime.now()
        LoggingUtil.display_text(f'PLRAgentManager __init__')
        if api_version is None or api_version.lower() not in VALID_API_VERSION_SET:
            raise ValueError(f'API Version is invalid; use one of these: {", ".join(VALID_API_VERSION_SET)}')
        self.api_version = api_version.lower()

        # Shared resources
        self.openai_controller_obj = self.__get_openai_controller()
        self.youtube_controller_obj = self.__get_youtube_controller()

        end_time = datetime.datetime.now()
        LoggingUtil.display_elapsed_time(start_time, end_time, 'PLRAgentManager __init__')
    
    def __get_openai_controller(self):
        """
        Get the OpenAI controller for llm queries, and vector store interactions.
        :return: OpenAI controller object
        """
        openai_controller_obj = None
        if self.api_version == VERSION_1:
            openai_controller_obj = OpenAIController()
        return openai_controller_obj
    
    def __get_youtube_controller(self):
        """
        Get the Youtube controller for interacting with the Youtube API.
        :return: Youtube controller object
        """
        youtube_controller_obj = None
        if self.api_version == VERSION_1:
            youtube_controller_obj = YoutubeController()
        return youtube_controller_obj

    def query_plr_agent(self, query_agent_payload):
        """
        Query the Personalized Learning Recommendations Agent.
        :param query_agent_payload: QueryAgentPayload object
        :return: response from the agent
        """
        LoggingUtil.display_text(f'PLRAgentManager query_plr_agent')
        start_time = datetime.datetime.now()
        success = False
        response = 'Failed to query Agent'
        video_results = []
        if self.api_version == VERSION_1:
            try:
                query = query_agent_payload.query
                response = 'No educational videos found'
                if query:
                    video_results = self.openai_controller_obj.extract_keywords(query, self.youtube_controller_obj)
                    response = 'Educational videos found'
                    success = True
                return {self.RESPONSE_KEY: response, self.SUCCESS_KEY: success, self.RECOMMENDATIONS_KEY: video_results}
            except Exception as e:
                LoggingUtil.display_text(f'Error in PLRAgentManager query_plr_agent: {str(e)}')
                raise e
            finally:
                end_time = datetime.datetime.now()
                LoggingUtil.display_elapsed_time(start_time, end_time, 'PLRAgentManager query_plr_agent')
        raise ValueError(self.get_not_implemented_api_msg('query_plr_agent'))
    
    def get_not_implemented_api_msg(self, method_name):
        """
        Get a message indicating that the method is not implemented for the current API version.
        :param method_name: Name of the method
        :return: not implemented message as string
        """
        return f'Method {method_name} is not implemented for API version {self.api_version}.'