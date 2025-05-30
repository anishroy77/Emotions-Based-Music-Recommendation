import cv2
import numpy as np
import tensorflow as tf
import webbrowser
import json
import time
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from youtubesearchpython import VideosSearch

# -------------------- Load Model & Setup --------------------
def load_emotion_model(model_path="emotion_model.h5"):
    return load_model(model_path)

model = load_emotion_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

emotion_music_mapping = {
    "Happy": ["party music", "dance hits"],
    "Sad": ["sad songs", "soft piano"],
    "Angry": ["rock music", "metal songs"],
    "Neutral": ["lofi music", "chill beats"],
    "Surprise": ["trending songs", "latest pop"],
    "Fear": ["calm music", "relaxing tunes"],
    "Disgust": ["instrumental", "classical music"]
}

uplifting_content_mapping = {
    "Sad": ["sad songs", "funny videos"],
    "Angry": ["calm music", "meditation videos"],
    "Fear": ["motivational songs", "inspirational talks"],
    "Disgust": ["chill beats", "stand-up comedy"]
}

history_file = "music_history.json"

# -------------------- Utility Functions --------------------
def save_history(emotion, recommendations):
    try:
        with open(history_file, "r") as file:
            history = json.load(file)
    except:
        history = []
    history.append({"emotion": emotion, "recommendations": recommendations})
    with open(history_file, "w") as file:
        json.dump(history, file, indent=4)

def load_history():
    try:
        with open(history_file, "r") as file:
            return json.load(file)
    except:
        return []

def get_recommendations(query, limit=3):
    try:
        search = VideosSearch(query, limit=limit).result()
        return [{"title": v["title"], "url": v["link"]} for v in search['result']]
    except:
        return []

def detect_emotion():
    cap = cv2.VideoCapture(0)
    detected_emotion = ""
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = img_to_array(roi.astype("float") / 255.0)
            roi = np.expand_dims(roi, axis=0)
            preds = model.predict(roi, verbose=0)[0]
            detected_emotion = emotion_labels[np.argmax(preds)]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        stframe.image(frame, channels="BGR")
        time.sleep(0.5)
        if detected_emotion:
            break

    cap.release()
    return detected_emotion

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="🎵 Emotion Music", layout="wide")

# Custom CSS for background and UI
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #360033, #0b8793);
        }
        .main {
            background: linear-gradient(to right, #360033, #0b8793);
            color: white;
        }
        footer {
            visibility: hidden;
        }
        .footer-style {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            background-color: rgba(0, 0, 0, 0.4);
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- Top Navigation --------------------
page = st.selectbox("Navigate", ["🏠 Home", "📜 History"])

# -------------------- Main UI --------------------
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 42px;'>🎵 Real-Time Emotion-Based Music Recommendation</h1>
        <p style='font-size: 20px;'>Detect your emotion and enjoy music tailored to your mood</p>
    </div>
""", unsafe_allow_html=True)

if page == "🏠 Home":
    mode = st.radio("Select Emotion Detection Mode", ["🎭 Detect Emotion via Webcam", "📷 Manually Select Emotion"])
    query = None

    if mode == "🎭 Detect Emotion via Webcam":
        if st.button("🎭 Detect Emotion"):
            query = detect_emotion()
    else:
        query = st.selectbox("📌 Select Emotion Manually", emotion_labels)
        if st.button("🎶 Get Music For This Emotion"):
            st.success(f"🎭 Selected Emotion: {query}")

    if query:
        if query in uplifting_content_mapping:
            option = st.radio("What would you prefer?", ["🎵 Same Mood Music", "🌞 Uplifting Content"])
            search_term = uplifting_content_mapping[query][0] if option.startswith("🎵") else uplifting_content_mapping[query][1]
        else:
            search_term = query

        recommendations = get_recommendations(search_term)
        save_history(query, recommendations)

        if recommendations:
            st.markdown("### 🎧 Recommendations:")
            for rec in recommendations:
                st.markdown(f"🔗 [{rec['title']}]({rec['url']})")
        else:
            st.warning("No recommendations found.")

elif page == "📜 History":
    st.subheader("📜 Recommendation History")
    history = load_history()
    if history:
        for entry in reversed(history):
            st.markdown(f"**🎭 Emotion/Query:** {entry['emotion']}")
            for rec in entry.get("recommendations", []):
                st.markdown(f"🔗 [{rec['title']}]({rec['url']})")
            st.markdown("---")
    else:
        st.info("No history found.")

# -------------------- Footer --------------------
st.markdown("""
<div class='footer-style'>
    Made with love ❤️
</div>
""", unsafe_allow_html=True)
