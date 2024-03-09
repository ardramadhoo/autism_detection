import streamlit as st
import numpy as np
import pickle
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the machine learning model
#with open('my_model.keras', 'rb') as model_file:
#    model = pickle.load(model_file)

# Function to predict autism
def predict_autism(image):
    # Preprocess the image (if needed)
    # Make prediction using the model
    prediction = np.random.choice(['Autistic', 'Non-autistic'])
    return prediction

# Streamlit UI
st.title('Autism Detection')

# Option to upload image or capture from webcam
option = st.radio("Select Option:", ("Upload Image", "Capture Live Picture"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        prediction = predict_autism(image)
        
        st.write(f'The person in the image is predicted to be: {prediction}')
elif option == "Capture Live Picture":
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            return frame

    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    if webrtc_ctx.video_transformer and hasattr(webrtc_ctx, 'video_frame'):
        prediction_button = st.button("Predict")
        if prediction_button:
            image = Image.fromarray(webrtc_ctx.video_frame)
            st.image(image, caption='Live Picture', use_column_width=True)
            st.write("")
            st.write("Classifying...")

            prediction = predict_autism(image)
            st.write(f'The person in the image is predicted to be: {prediction}')
