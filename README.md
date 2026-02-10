<div align="center">

# 👚 Fashion MNIST Image Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)

**A deep learning-powered tool for classifying fashion items from the Fashion MNIST dataset.**

[View Demo](https://your-streamlit-app-link.streamlit.app/) 

</div>

---

## 📖 Project Overview

This application provides a web-based tool for classifying fashion items from the Fashion MNIST dataset. It leverages a deep learning model trained to recognize different categories of clothing and accessories, allowing users to upload an image and receive a prediction with confidence scores. This project demonstrates the integration of machine learning models with interactive web interfaces, featuring custom styling for a more engaging user experience.

---

## 🚀 Key Components

| Component | Description |
| :--- | :--- |
| **🧠 Neural Network** | A sequential Keras model (`fashion_mnist_model.keras`) optimized for 10-class image classification. |
| **🏷️ Class Names** | A `pickle` file (`class_names.pkl`) containing the human-readable labels for the fashion categories. |
| **💻 Streamlit UI** | An interactive web dashboard (`app.py`) for real-time image input, prediction, and visualization. |

---

## 🛠️ Features & Inputs

The model analyzes **28x28 grayscale images** to predict the fashion item category. Users can upload an image, and the application will provide a classification with a confidence score.

<div align="center">

| Feature | Description |
| :--- | :--- |
| **Image Input** | Accepts JPG, JPEG, and PNG image files, processed into 28x28 grayscale for classification. |
| **Classification Output** | Displays the predicted fashion item class (e.g., 'T-shirt/top', 'Ankle boot') and its confidence level. |
| **Custom Background** | Features a classy linear gradient background (`linear-gradient(to right, #2c3e50, #4a69bd)`) for aesthetic appeal. |
| **Styled Drag-and-Drop** | An intuitive image upload area with a complementary styled linear gradient (`linear-gradient(to right, #eceff1, #e0e7f1)`) and rounded corners. |

</div>
