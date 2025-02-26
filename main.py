__author__ = 'cochoa, anjali, thiru, fernando'

import logging
import sys

from flask import Flask, jsonify

sys.path.insert(0, "bin/action/data/")
from summary_data_action import summary_data_action                 # Data API
from category_data_action import category_data_action               # Data API
from push_data_action import push_data_action                       # Data API
from feedback_data_action import feedback_data_action               # Data API
from delete_data_action import delete_data_action                   # Data API
from word_cloud_data_action import word_cloud_data_action           # Data API
from emotion_data_action import emotion_data_action                 # Data API
from intent_data_action import intent_data_action                   # Data API
from sentiment_data_action import sentiment_data_action             # Data API
from categories_timeline_action import categories_timeline_action   # Data API

sys.path.insert(0, "bin/action/data_modification/")
from data_modification_action import data_modification_action       # Data Modification API

sys.path.insert(0, "bin/action/model_mapping/")
from model_mapping_action import model_mapping_action               # Model to data set mapping API
from model_stack_action import model_stack_action                   # Model stack API
from configure_settings_action import configure_settings_action     # Settings API

sys.path.insert(0, "bin/action/mysql_table_mapping")
from mysql_table_mapping_action import mysql_table_mapping_action   # Mysql table mapping API

sys.path.insert(0, "bin/action/reclassification")
from reclassification_action import reclassification_action         # Mysql historical reclassification API

sys.path.insert(0, "bin/action/monkey_learn/")
from monkey_learn_usage_action import monkey_learn_usage_action     # MonkeyLearn usage API - deprecated

sys.path.insert(0, "bin/action/nlp_query/")
from nlp_query_usage_action import nlp_query_usage_action           # NLP query usage API

sys.path.insert(0, "bin/action/chatbot/")
from query_chatbot_action import query_chatbot_action               # Query chatbot API

sys.path.insert(0, "bin/action/insight/")
from insight_action import insight_action                           # Insight Platform-related API

sys.path.insert(0, "bin/action/action_plan/")
from action_plan_action import action_plan_action                   # Action Plan API

sys.path.insert(0, "bin/action/emotion_detection/")
from emotion_detection_action import emotion_detection_action       # Emotion Detection API

sys.path.insert(0, "bin/action/ta_pipeline_report/")
from ta_pipeline_report_action import ta_pipeline_report_action     # TA Pipeline Report API

sys.path.insert(0, 'bin/action/help_center/')
from help_center_action import help_center_action                   # Help Center API

sys.path.insert(0, 'bin/action/game_plans/')
from game_plans_action import game_plans_action                     # Game Plans API

sys.path.insert(0, 'bin/action/decision_ai/')
from decision_ai_action import decision_ai_action                   # Decision AI API

sys.path.insert(0, 'bin/action/vision_ai/')
from vision_ai_action import vision_ai_action                       # Vision AI API

sys.path.insert(0, "bin/action/speech_ai/")
from speech_ai_action  import speech_ai_action                      # Speech AI API

sys.path.insert(0, "bin/action/huddle_harmony/")
from huddle_harmony_action import huddle_harmony_action             # Huddle Harmony API - AI Meeting Assistant

sys.path.insert(0, "bin/action/document_comparison")                # Document Comparison API
from document_comparison_action import document_comparison_action

sys.path.insert(0, "bin/action/plr_agent/")                        # PLR Agent API
from plr_agent_action import plr_agent_action


app = Flask(__name__)
app.register_blueprint(summary_data_action)
app.register_blueprint(category_data_action)
app.register_blueprint(push_data_action)
app.register_blueprint(feedback_data_action)
app.register_blueprint(delete_data_action)
app.register_blueprint(word_cloud_data_action)
app.register_blueprint(emotion_data_action)
app.register_blueprint(intent_data_action)
app.register_blueprint(sentiment_data_action)
app.register_blueprint(categories_timeline_action)
app.register_blueprint(data_modification_action)
app.register_blueprint(model_mapping_action)
app.register_blueprint(model_stack_action)
app.register_blueprint(mysql_table_mapping_action)
app.register_blueprint(reclassification_action)
app.register_blueprint(configure_settings_action)
app.register_blueprint(monkey_learn_usage_action)
app.register_blueprint(nlp_query_usage_action)
app.register_blueprint(query_chatbot_action)
app.register_blueprint(insight_action)
app.register_blueprint(action_plan_action)
app.register_blueprint(emotion_detection_action)
app.register_blueprint(ta_pipeline_report_action)
app.register_blueprint(help_center_action)
app.register_blueprint(game_plans_action)
app.register_blueprint(decision_ai_action)
app.register_blueprint(vision_ai_action)
app.register_blueprint(speech_ai_action)
app.register_blueprint(huddle_harmony_action)
app.register_blueprint(document_comparison_action)
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
