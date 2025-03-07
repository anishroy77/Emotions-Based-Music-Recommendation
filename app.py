import cv2
import numpy as np
import tensorflow as tf
import webbrowser
import json
import time
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from pytube import Search


def load_emotion_model(model_path="emotion_model.h5"):
    return load_model(model_path)


# Load the pre-trained emotion model
model = load_emotion_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Mapping emotions to music genres
emotion_music_mapping = {
    "Happy": ["party music", "dance hits"],
    "Sad": ["sad songs", "soft piano"],
    "Angry": ["rock music", "metal songs"],
    "Neutral": ["lofi music", "chill beats"],
    "Surprise": ["trending songs", "latest pop"],
    "Fear": ["calm music", "relaxing tunes"],
    "Disgust": ["instrumental", "classical music"]
}

# History tracking file
history_file = "music_history.json"


def save_history(emotion, video_url):
    try:
        with open(history_file, "r") as file:
            history = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append({"emotion": emotion, "video_url": video_url})

    with open(history_file, "w") as file:
        json.dump(history, file, indent=4)


def get_music_recommendation(emotion):
    queries = emotion_music_mapping.get(emotion, ["popular songs"])
    for query in queries:
        search_results = Search(query).results
        if search_results:
            video_url = search_results[0].watch_url
            save_history(emotion, video_url)
            return video_url
    return None


def detect_emotion():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return None

    ret, frame = cap.read()
    cap.release()
    if not ret:
        st.error("Error: Could not capture frame.")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected. Try again.")
        return None

    x, y, w, h = faces[0]
    roi = gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    preds = model.predict(roi)[0]
    emotion = emotion_labels[np.argmax(preds)]
    return emotion


# Streamlit UI
st.title("Real-Time Emotion-Based Music Recommendation")
st.write("Click the button below to detect your emotion and get a music recommendation.")

if st.button("Detect Emotion"):
    emotion = detect_emotion()
    if emotion:
        st.success(f"Detected Emotion: {emotion}")
        video_url = get_music_recommendation(emotion)
        if video_url:
            st.write(f"Recommended Music: [Click Here]({video_url})")
        else:
            st.warning("No music recommendation found.")