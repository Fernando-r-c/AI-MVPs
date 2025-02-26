__author__ = 'fernando'

import logging
import sys

from flask import Flask, jsonify

sys.path.insert(0, "bin/action/plr_agent/")                        # PLR Agent API
from plr_agent_action import plr_agent_action

app = Flask(__name__)
app.register_blueprint(plr_agent_action)

ERROR_STATUS = 'error'

@app.errorhandler(400)
def bad_request_error(e):
    """
    Handle bad request errors
    :param e: exception
    :return: bad request error message
    """
    message = 'An error occurred during a request: {}'.format(e)
    logging.exception(message)
    return __create_api_response(ERROR_STATUS, message, 400)


@app.errorhandler(404)
def not_found_error(e):
    """
    Handle not found errors
    :param e: exception
    :return: not found error message
    """
    message = 'An error occurred during a request: {}'.format(e)
    logging.exception(message)
    return __create_api_response(ERROR_STATUS, message, 404)


@app.errorhandler(500)
def server_error(e):
    """
    Handle server errors
    :param e: exception
    :return: server error message
    """
    message = 'An error occurred during a request: {}'.format(e)
    logging.exception(message)
    return __create_api_response(ERROR_STATUS, message, 500)


@app.errorhandler(502)
def bad_gateway_error(e):
    """
    Handle bad gateway errors
    :param e: exception
    :return: server error message
    """
    message = 'An error occurred during a request: {}'.format(e)
    logging.exception(message)
    return __create_api_response(ERROR_STATUS, message, 502)


def __create_api_response(status, message, status_code):
    """
    Create an API response with status, message, status code and submission token
    :param status: 'error' or 'success'
    :param message: reason for success or failure
    :param status_code: status code
    :return: json api response with status code
    """
    payload = {'status': status, 'message': message}
    api_response = jsonify(payload)
    api_response.status_code = status_code
    return api_response


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
    # app.run(host='127.0.0.1', port=8080, debug=False, threaded=True)
    # app.run(host='127.0.0.1', port=8080, debug=True)
