__author__ = 'fernando'

import logging

from flask import Flask, jsonify
import streamlit as st
import requests
from streamlit_lottie import st_lottie

def load_lottie_url(url):
    """
    Get the Lottie JSON from the URL
    :param url: URL to get the Lottie JSON from
    :return: Lottie JSON
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_education = load_lottie_url("https://lottie.host/895421d5-e868-430b-9637-5417a3f66286/pGMTaxPOCo.json")


def main():  
    st.sidebar.title("Navigation")
    st.sidebar.info("Use this app to receive personalized learning video recommendations.")
    st.sidebar.markdown("---")

    st.title(":mortar_board: Personalized Learning Recommendations")
    st_lottie(lottie_education, height=300, key="education")

    st.subheader("Tell us about your learning goals")
    user_input = st.text_input(
        "What do you want to learn today?",
        placeholder="e.g., Python programming, Data Science, Machine Learning",
    )

    
    # Query the agent if input is provided
    if st.button("Get Recommendations"):
        if user_input.strip():
            st.info("Fetching the best learning resources for you... Please wait.")
            plr_agent = 'http://localhost:5000/eliza/plr-agent/api/v1/query'
            try:
                response = requests.post(plr_agent, json={'query': user_input}).json()
                logging.info(f"Response received: {response}")

                if response.get('recommendations'):
                    st.success("Here are your personalized recommendations:")
                    for idx, video in enumerate(response['recommendations'], start=1):
                        video_id = video['link'].split('v=')[-1]
                        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.image(thumbnail_url, use_container_width=True)
                        with col2:
                            st.write(f"### {idx}. {video['title']}")
                            st.write(f"{video['description']}")
                            st.markdown(f"[Watch Video]({video['link']})")
                        st.markdown("---")
                else:
                        st.warning("No recommendations found. Please try a different query.")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the recommendation agent. Error: {e}")
        else:
            st.error("Please enter a learning goal before requesting recommendations.")

if __name__ == "__main__":
    main()