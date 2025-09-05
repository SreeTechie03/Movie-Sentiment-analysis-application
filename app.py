from keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

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
st.title("🎬 Movie Sentiment Analysis Application")
st.write("Enter a movie review below and get the predicted sentiment:")

review = st.text_area("Movie Review")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review first!")
    else:
        sentiment = predictive_system(review)
        st.success(f"Prediction: {sentiment}")
