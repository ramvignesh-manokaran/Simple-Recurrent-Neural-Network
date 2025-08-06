# Step 1: Import libraries and load the model

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

## Load the trained model
model = load_model('simple_rnn_imdb.h5')
model.summary()


# Step 2 : Helper Functions

## Function to decode reviews
def decode_review(encoded_review):
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    return decoded_review

## Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Prediction function

def predict_sentiment(review):
  preprocessed_input = preprocess_text(review)

  prediction = model.predict(preprocessed_input)

  sentiment = 'Positive' if prediction > 0.5 else 'Negative'

  return sentiment, prediction[0][0]

## Streamlit app
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")

#user input
user_input = st.text_area("Enter your movie review:")

if st.button("Classify"):
    preprocess_input = preprocess_text(user_input)
    sentiment, confidence = predict_sentiment(preprocess_input)


    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence: {confidence:.2f}")
else:
    st.write("Please enter a movie review and click 'Classify' to see the sentiment analysis.")
