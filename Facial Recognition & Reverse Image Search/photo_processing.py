__author__ = 'fernando'

import cv2
import json
import os
import numpy as np
import pyimgur
import requests
import sys

from io import BytesIO
from PIL import Image
from serpapi import GoogleSearch

sys.path.insert(0, 'bin/controllers/')
from openai_controller import OpenAIController

prototxt_path = "bin/resources/vision_ai/deploy.prototxt"
model_path = "bin/resources/vision_ai/res10_300x300_ssd_iter_140000.caffemodel"

def capture_photo(photo):
    """
    Upload a photo from a file
    :param photo: The object of the photo
    """
    image = cv2.imread(photo)
    cv2.imshow("Photo", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces(photo):
    """
    Detect faces in a photo
    :param photo: The object of the photo
    :return: The path of the photo with the faces detected and the faces detected
    """
    image = cv2.imread(photo)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Using Haar cascades for face detection")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if faces == ():
        print("No faces detected.")
        return detect_faces_dnn(photo)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)    
    detected_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return detected_image, faces

def search_similar_images(photo):
    """
    Search for similar images on the internet
    :param photo: The object of the photo
    :return: list of search results of similar images to the input photo
    """
    serpapi_api_key = "85dff4012e1c33b171427ed9c42b70d69515f864b896b454b213d95484b50139"
    imgur_client_id = "69025f67f7d9c2f"
    search_results = []
    absolute_image_path = os.path.join(os.getcwd(), photo)
    im = pyimgur.Imgur(imgur_client_id)
    uploaded_image = im.upload_image(absolute_image_path)
    params = {
        "engine": "google_lens",
        "url": uploaded_image.link,
        "api_key": serpapi_api_key
    }
    search = GoogleSearch(params)
    if search:
        results = search.get_dict()
        print(f'Results: {json.dumps(results, indent=2)}')
        visual_matches = results["visual_matches"]
        for visual_match in visual_matches:
            search_results.append({
                "title": visual_match["title"],
                "link": visual_match["link"],
                "thumbnail": visual_match["thumbnail"]
            })
    print(f"Search results: {search_results}")
    return search_results

def detect_faces_dnn(photo):
    print("Using DNN for face detection")
    net = cv2.dnn.readNetFromCaffe(
        prototxt_path, model_path
    )

    image = cv2.imread(photo)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            faces.append((x, y, x2, -x, y2 - y))
            cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)

    detected_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return detected_image, faces

def crop_face(image_path, face_coordinates):
    """
    Crop the face from the image using the provided coordinates.
    :param image_path: Path to the original image
    :param face_coordinates: Coordinates of the detected face
    :return: Cropped face image
    """
    image = Image.open(image_path)
    for (x, y, w, h) in face_coordinates:
        cropped_face = image.crop((x, y, x + w, y + h))
        return cropped_face
    
def summarize_title_and_links(titles_and_links_result):
    """
    Summarize the titles and links
    :param titles_and_links: List of titles and links
    :return: Summary of titles and links
    """
    #use the OpenAI Controller llm to summarize the titles and links
    print(f"titles_and_links_result: {titles_and_links_result}")
    openai_controller = OpenAIController()
    llm = openai_controller.llm_chat
    prompt = "Summarize the following titles and links into 3 sentences:\n"
    for title_and_link in titles_and_links_result:
        prompt += f"Title: {title_and_link['title']}, Link: {title_and_link['link']}\n"
    response = llm.predict(prompt)
    print(f"Response: {response}")
    return response

def resize_image(image_url, size=(150, 150)):
    """
    Resize the image from the given URL to fit within the specified size while maintaining aspect ratio.
    :param image_url: URL of the image to resize
    :param size: Tuple specifying the size (width, height)
    :return: Resized image
    """
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image.thumbnail(size, Image.LANCZOS)
    return image
    


if __name__ == "__main__":
    photo_path = "detected_faces.jpg"
    search_results = search_similar_images(photo_path)
    for result in search_results:
        print(f"Title: {result['title']}, Link: {result['link']}")
