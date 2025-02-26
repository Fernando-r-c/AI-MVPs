__author__ = 'fernando, ichchitaa'

import datetime
import sys

from flask import Blueprint, request, jsonify
from flask_cors import CORS, cross_origin

sys.path.insert(0, "bin/action/")
from error_handler_action import ErrorHandlerAction

sys.path.insert(0, 'bin/action/plr_agent/')
from query_agent_payload import QueryAgentPayload

sys.path.insert(0, 'bin/apps/plr_agent/')
from plr_agent_manager import PLRAgentManager

sys.path.insert(0, 'bin/common/utils/')
from logging_util import LoggingUtil

from io import BytesIO


plr_agent_action = Blueprint('plr_agent_action', __name__)
CORS(plr_agent_action, resources={r"/eliza/*": {"origins": "*"}}, allow_headers='Content-Type')



@plr_agent_action.route('/eliza/plr-agent/api/<string:api_version>/query', methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def query_agent(api_version):
    """
    Query the Personalized Learning Recommendations Agent
    :param api_version: api version to use
    :return: response
    """
    start_time = datetime.datetime.now()
    try:
        query_agent_payload = QueryAgentPayload(request.json)
        query_agent_payload.display_contents()
        if query_agent_payload.has_error():
            response = __create_400_error_response(query_agent_payload.error_message)
        else:
            response_dict = PLRAgentManager(api_version).query_plr_agent(query_agent_payload)
            status_code = __get_status_code(response_dict)
            response = __create_api_response(status_code, response_dict)
            LoggingUtil.display_text(f'PLRAgentAction.query_agent: response - {response_dict}')
    except Exception as e:
        LoggingUtil.display_text(f'PLRAgentAction.query_agent: Exception - {e}')
        response = __create_500_error_response(str(e))
    finally:
        end_time = datetime.datetime.now()
        LoggingUtil.display_elapsed_time(start_time, end_time, 'PLRAgentAction.query_agent')
        return response
    

def __create_500_error_response(error_message):
    payload = {'success': False, 'response': error_message}
    return __create_api_response(500, payload)


def __create_400_error_response(error_message):
    payload = {'success': False, 'response': error_message}
    return __create_api_response(400, payload)

def __create_api_response(api_status_code, payload):
    """
    Create an API response with success, response, and status code.
    :param api_status_code: status code
    :param payload: response payload
    :return: json api response with status code
    """
    api_response = jsonify(payload)
    api_response.status_code = api_status_code
    return api_response


def __get_status_code(response_dict):
    """
    Get the status code based on the success flag in the response dictionary.
    :param response_dict: response dictionary
    :return: status code
    """
    return 200 if response_dict['success'] else 404
    
