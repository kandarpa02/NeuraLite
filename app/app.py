import streamlit as st
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import preprocess_image
from src.utils import predict


def upload_and_predict(model_file):
    st.title("MNIST Digit Classifier")
    st.write("Upload an image to predict the digit!")

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded_file:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        try:
            # Load the model
            with open(model_file, 'rb') as file:
                model = pickle.load(file)
            W, b = model['weights'], model['bias']

            # Preprocess the uploaded image
            img_flattened = preprocess_image(uploaded_file)

            # Make prediction
            y_pred = predict(img_flattened, W, b)

            # Display the prediction
            st.write(f"Prediction: {y_pred[0]}")
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")


# Path to the serialized model
model_file = "/home/kandarpa-sarkar/Desktop/NeuraLite/model/trained_model.pkl"  # Update with the actual path to your .pkl file

# Run the Streamlit app
if __name__ == "__main__":
    upload_and_predict(model_file)
