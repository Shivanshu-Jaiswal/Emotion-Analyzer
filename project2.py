import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

# -----------------------------
# Preprocessing
# -----------------------------

stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()

    text = re.sub(r"\d+", "", text)

    text = text.encode("ascii", "ignore").decode()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# -----------------------------
# Load Model
# -----------------------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# -----------------------------
# Emotion Mapping
# -----------------------------

emotion_map = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

emoji = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Emotion Analyzer",
    page_icon="🎭"
)

st.title("🎭 Emotion Analyzer")

st.write(
    "Enter text and detect the emotion using Machine Learning."
)

text = st.text_area(
    "Enter Text",
    height=150,
    placeholder="Example: I am very happy today"
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Emotion"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:
        clean_text = preprocess(text)

        vector = vectorizer.transform([clean_text])

        prediction = int(model.predict(vector)[0])

        emotion = emotion_map[prediction]

        confidence = max(
            model.predict_proba(vector)[0]
        ) * 100

        st.success(
            f"{emoji[emotion]} Emotion: {emotion.capitalize()}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        with st.expander("Processed Text"):
            st.write(clean_text)