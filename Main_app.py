import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Add CSS for custom background
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(to right, #2c3e50, #4a69bd);
        background-size: cover;
    }
    /* Targeting the file uploader input element */
    .st-emotion-cache-1uixx5h {
        background-image: linear-gradient(to right, #f0e3ff, #e0f2ff);
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('fashion_mnist_model.keras')
    return model
model = load_model()

# Load the class names
@st.cache_data
def load_class_names():
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    return class_names

class_names = load_class_names()

st.title('Fashion MNIST Image Classifier')
st.write('Upload an image of a clothing item.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    image = image.convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    # Get the confidence score
    confidence = tf.nn.softmax(predictions[0])[predicted_class_index] * 100

    # Display the prediction
    st.success(f"Prediction: {predicted_class_name}")
    st.write(f"Confidence: {confidence:.2f}%")

