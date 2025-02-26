__author__ = 'fernando'

import sys

sys.path.insert(0, 'bin/apps/common/')
from base_payload import BasePayload

sys.path.insert(0, 'bin/common/utils/')
from logging_util import LoggingUtil


class QueryAgentPayload(BasePayload):

    QUERY_KEY = 'query'

    def __init__(self, request_json):
        """
        Initialize the QueryAgentPayload object.
        :param request_json: request JSON
        """
        super(QueryAgentPayload, self).__init__(request_json)
        self.query = self.valid_payload.get(self.QUERY_KEY)
        self.error_message = self.__get_error_message()
        
    def __get_error_message(self):
        """
        Get the error message.
        :return: error message (empty if no error)
        """
        message = 'JSON request body is empty'
        if self.valid_payload:
            message = ''
            message += self.validate_param(self.query, self.QUERY_KEY)
        return message


    def display_contents(self):
        LoggingUtil.display_text(f'\n Query Agent Payload')
        LoggingUtil.display_text('- - - - - - - - -')
        super(QueryAgentPayload, self).display_contents()
        LoggingUtil.display_text(f'query: {self.query}')
