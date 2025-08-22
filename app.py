
from keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr

# Load model + tokenizer
model = load_model("model.h5")
tokenizer = joblib.load("tokenizer.pkl")

def predictive_system(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequences = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequences)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment

# Gradio app
title = "🎬 Movie Sentiment Analysis Application"
app = gr.Interface(
    fn=predictive_system,
    inputs="textbox",
    outputs="text",
    title=title
)

if __name__ == "__main__":
    app.launch()
