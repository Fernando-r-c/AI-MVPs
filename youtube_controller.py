__author__ = 'fernando'

import os
import sys

from googleapiclient.discovery import build

sys.path.insert(0, "bin/common/")
from ini_config_reader import read_ini_config_section, DEFAULT_CONFIG_FILE

sys.path.insert(0, 'bin/common/utils/')
from logging_util import LoggingUtil

class YoutubeController:
    """
    Class to interact with Youtube API
    """

    def __init__(self):
        """
        Constructor
        """
        self.api_key = os.getenv('YOUTUBE_API_KEY')
        LoggingUtil.display_text(f'Youtube API Key: {self.api_key}')
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)

    def search_educational_videos(self, query, max_results=5):
        """
        Search for educational videos on Youtube
        :param query: query to search for
        :param max_results: maximum number of results to return
        :return: list of video ids
        """
        search_response = self.youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=max_results,
            type='video',
            videoCategoryId='27', # Education category
        ).execute()

        videos = []
        for search_result in search_response.get('items', []):
            if search_result['id']['kind'] == 'youtube#video':
                result_dict ={
                    'title': search_result['snippet']['title'],
                    'link': 'https://www.youtube.com/watch?v=' + search_result['id']['videoId'],
                    'description': str(search_result['snippet']['description']).replace("'", "/'") # remove single quotes from description
                }
                videos.append(result_dict)
        return videos