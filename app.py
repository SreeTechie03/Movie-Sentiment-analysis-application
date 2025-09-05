from keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

# Page config
st.set_page_config(page_title="Movie Sentiment Analyzer 🎬", page_icon="🎭", layout="centered")

# Custom CSS for background + styling
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #4CAF50;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #45a049;
            color: #fff;
        }
    </style>
""", unsafe_allow_html=True)

# Load model + tokenizer
model = load_model("model.h5")
tokenizer = joblib.load("tokenizer.pkl")

# Prediction function
def predictive_system(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequences = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequences)
    sentiment = "😊 Positive" if prediction[0][0] > 0.5 else "😞 Negative"
    return sentiment

# Streamlit UI
st.title("🎬 Movie Sentiment Analysis")
st.markdown("### ✨ Enter a movie review and let AI predict the sentiment!")

review = st.text_area("📝 Type your movie review here:")

if st.button("🔍 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review first!")
    else:
        sentiment = predictive_system(review)
        if "Positive" in sentiment:
            st.success(f"🌟 Prediction: {sentiment}")
        else:
            st.error(f"💔 Prediction: {sentiment}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using **Keras + Streamlit**")
