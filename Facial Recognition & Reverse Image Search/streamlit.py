__author__ = 'fernando'

import streamlit as st
import requests

from streamlit_lottie import st_lottie
from photo_processing import capture_photo, detect_faces, search_similar_images, crop_face, summarize_title_and_links, resize_image

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

lottie_facial_recognition = load_lottie_url("https://lottie.host/821ddc87-b719-4a71-a565-9cc578a6e5b5/8WW0rFFrDx.json")

def main():
    """
    Main function
    """
    st.set_page_config(layout="wide")
    with st.sidebar:
        st.title("Photo Processing App")
        st_lottie(lottie_facial_recognition, height=300, key="facial-recognition")
        st.header("Summary")
        st.write("This MVP demonstrates facial recognition by capturing a photo, detecting faces, and searching for similar images using web sources.")
    col_buttons, col_display = st.columns([1, 3])

    with col_buttons:
        st.header("Upload a Photo")
        #upload_button = st.button("Upload Photo")
        #capture_button = st.button("Capture Photo")

    with col_display:
        col_image, col_results = st.columns([1, 2])
        # Uploaded photo
        uploaded_photo = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
        if uploaded_photo:
            with open("uploaded_photo.jpg", "wb") as f:
                f.write(uploaded_photo.getbuffer())
            col_image.image(uploaded_photo, caption="Uploaded Image", use_container_width=True)
            detected_faces_image, faces = detect_faces("uploaded_photo.jpg")
            if len(faces) > 0:
                col_results.image(detected_faces_image, caption="Detected faces", use_container_width=True)
                for face_coordinates in faces:
                    cropped_face = crop_face("uploaded_photo.jpg", [face_coordinates])
                    cropped_face = cropped_face.convert("RGB")  # Convert to RGB mode
                    cropped_face_path = "cropped_face.jpg"
                    cropped_face.save(cropped_face_path)
                    search_results = search_similar_images(cropped_face_path)
                    title_and_links = []
                    for i in range(0, min(len(search_results), 12), 3):
                        cols = st.columns(3)
                        for j in range(3):
                            if i + j < min(len(search_results), 12):
                                result = search_results[i + j]
                                with cols[j]:
                                    title_and_links.append(result)

                    #summarize
                    if len(title_and_links) > 0: 
                        summary = summarize_title_and_links(title_and_links)
                        st.header("Summary of All Titles/Links")
                        st.write(summary)
                        st.header("Image Search Results")
                        for i in range(0, len(title_and_links), 3):
                            cols = st.columns(3)
                            for j in range(3):
                                if i + j < len(title_and_links):
                                    result = title_and_links[i + j]
                                    with cols[j]:
                                        st.write(f"[{result['title']}]({result['link']})")
                                        resized_thumbnail = resize_image(result["thumbnail"])
                                        st.image(resized_thumbnail, caption=result["title"], use_container_width=True)
                    else:
                        st.write("No search results found.")
            else:
                st.write("No faces detected.")  

if __name__ == "__main__":
    main()
