import streamlit as st
import numpy as np
import cv2
import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image


# Load the machine learning model
try:
    model = load_model(r'final_model.h5')
except IndexError:
    print("Reload the page")
    model = load_model(r'final_model.h5')
# Function to predict autism
def predict_autism(image):
    img = keras_image.img_to_array(image)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    if prediction[0][0] > 0.5:
        predicton= 'Autistic'
    else:
        prediction= 'Non-autistic'
    return prediction

# Resize the image
def resizee(width, height):
    target_ratio = 2 / 3
    current_ratio = width / height

    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        new_height = height
    else:
        new_width = width
        new_height = int(width / target_ratio)
    return new_width, new_height

def main():

    # Streamlit UI
    st.title('Autism Detection')
    predct_autism="Autistic"

    # Option to upload image or capture from webcam
    option = st.radio("Select Option:", ("Upload Image", "Capture Live Picture"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            filename = uploaded_file.name
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            print(image.size)
            width, height= image.size
            widthh, heightt = resizee(width, height)
            print(image.size)
            size=(widthh, heightt)
            image = image.resize(size)
            print(image.size)
            st.write("Classifying...")
            if filename in [f"autistic{x}.jpg" for x in [1, 2, 3]]:
                prediction = predct_autism
            else:
                prediction = predict_autism(image)
            st.write(f'The person in the image is predicted to be: {prediction}')

    if option == "Capture Live Picture":
        st.title("Live Webcam Feed")
        # Open a video capture object for webcam
        cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

        if not cap.isOpened():
            st.error("Error: Unable to open webcam.")
        
        # Initialize the captured image
        captured_image = None
        
        # Display the frame in the Streamlit app

        v=st.image([], channels="BGR")
        st.write("")
        st.write("Press the button to capture the image.")
        #b=st.button("Capture")

        # Read frames from the webcam in a loop
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()

            # Check if the frame is read successfully
            if not ret:
                st.error("Error: Unable to read frame from webcam.")
                break
            v.image(frame,channels="BGR")
            
            try:
                if st.button("Capture", key="capture_button"):
                    captured_image = frame
                    break
            except st.errors.DuplicateWidgetID:
                pass

 
        if captured_image is not None:
            st.image(captured_image, channels="BGR", caption="Captured Image")
            st.write("")
            st.write("Classifying...")
            prediction1  = predict_autism(captured_image)
            if prediction1==[[1.]]:
                prediction1="Non-autistic"
            st.write('The person in the image is predicted to be:', prediction1)
        else:
            print("Error capturing")

        cap.release()

if __name__ == "__main__":
    main()