__author__ = 'fernando'

import sys

class QueryAgentPayload():

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
            message += self.__validate_param(self.query, self.QUERY_KEY)
        return message
    
    def __validate_param(self, param, param_name):
        """
        Validate the parameter value for the given parameter name.
        :param param: parameter
        :param param_name: parameter name
        :return: error message (empty if no error)
        """
        message = ''
        if param is None:
            message += f'`{param_name}` is missing. '
        elif isinstance(param, str) and param.strip() == '':
            message += f'`{param_name}` is empty. '
        return message
    



    def display_contents(self):
        print(f'\n Query Agent Payload')
        print('- - - - - - - - -')
        super(QueryAgentPayload, self).display_contents()
        print(f'query: {self.query}')
