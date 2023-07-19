import streamlit as st
import tensorflow as tf
import h5py
import numpy as np
from PIL import Image
from keras.models import load_model

model = load_model(r"C:\Users\gangw\OneDrive\Desktop\Self Driving\traffic_signal.hdf5")

st.markdown(
    """
    <style>
    body { 
        background-color: #f0f0f0; /* Set your desired background color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

class_names = ['Speed limit (20km/h)',
               'Speed limit (30km/h)',
               'Speed limit (50km/h)',
               'Speed limit (60km/h)',
               'Speed limit (70km/h)',
               'Speed limit (80km/h)',
               'End of speed limit (80km/h)',
               'Speed limit (100km/h)',
               'Speed limit (120km/h)',
               'No passing',
               'No passing veh over 3.5 tons',
               'Right-of-way at intersection',
               'Priority road',
               'Yield',
               'Stop',
               'No vehicles',
               'Veh > 3.5 tons prohibited',
               'No entry',
               'General caution',
               'Dangerous curve left',
               'Dangerous curve right',
               'Double curve',
               'Bumpy road',
               'Slippery road',
               'Road narrows on the right',
               'Road work',
               'Traffic signals',
               'Pedestrians',
               'Children crossing',
               'Bicycles crossing',
               'Beware of ice/snow',
               'Wild animals crossing',
               'End speed + passing limits',
               'Turn right ahead',
               'Turn left ahead',
               'Ahead only',
               'Go straight or right',
               'Go straight or left',
               'Keep right',
               'Keep left',
               'Roundabout mandatory',
               'End of no passing',
               'End no passing veh > 3.5 tons']  # Replace with your actual class names

st.title("Traffic Sign Classification")
st.write("Upload an image and let the model predict its class.")


# Get the uploaded image file
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Display the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    image = image.resize((30,30))
    classification_button = st.button("classfiy")

    if classification_button:
        image = np.array(image)
        try:
            pred_probs = model.predict(np.expand_dims(image, axis=0))
            pred = np.argmax(pred_probs)
            sign = class_names[pred]
            st.write(sign)
        except Exception as e:
            st.error("An error occured during classification: "+str(e))
            
            
# run the line given below in terminal to run the code in local host
# streamlit run C:\Users\gangw\DS\traffic_siganl_deploy_streamlit.py 

