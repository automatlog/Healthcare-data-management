import os
import json
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import cv2
import numpy as np

# Initialize NLP resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize a dictionary to store patient data
patient_data = {}

# Function to add patient information
def add_patient(patient_id, name, medical_records, images):
    patient_data[patient_id] = {
        "name": name,
        "medical_records": medical_records,
        "images": images
    }

# Function to search for patients based on NLP
def search_patients(query):
    # Tokenize and remove stopwords from the query
    query_tokens = [word.lower() for word in word_tokenize(query) if word.isalnum() and word.lower() not in stop_words]

    # Calculate TF-IDF vectors for patient medical records
    tfidf_vectorizer = TfidfVectorizer()
    medical_records = [patient["medical_records"] for patient in patient_data.values()]
    tfidf_matrix = tfidf_vectorizer.fit_transform(medical_records)

    # Calculate cosine similarity between the query and medical records
    query_vector = tfidf_vectorizer.transform([' '.join(query_tokens)])
    similarities = cosine_similarity(query_vector, tfidf_matrix)

    # Find the most similar patient(s)
    results = []
    for i, score in enumerate(similarities[0]):
        if score > 0.2:  # Adjust the threshold as needed
            results.append((list(patient_data.keys())[i], score))

    return results

# Function to view patient information
def view_patient(patient_id):
    if patient_id in patient_data:
        return patient_data[patient_id]
    else:
        return None

# Function to add patient images
def add_patient_images(patient_id, image_paths):
    if patient_id in patient_data:
        patient_data[patient_id]["images"].extend(image_paths)
    else:
        return "Patient not found."

# Function to analyze medical images using AI models (e.g., image classification, segmentation, etc.)
def analyze_images(image_paths):
    results = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        # Add your AI image analysis code here
        # For example, you can use pre-trained deep learning models like ResNet, VGG, etc.
        # to classify or analyze medical images.
        # You can also use segmentation models for specific tasks.

        # Placeholder code to simulate analysis results
        analysis_result = {"image_path": image_path, "result": "Normal"}
        results.append(analysis_result)

    return results

# Save and load patient data to/from a JSON file for persistence
def save_patient_data(filename):
    with open(filename, 'w') as json_file:
        json.dump(patient_data, json_file)

def load_patient_data(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as json_file:
            return json.load(json_file)
    else:
        return {}

# Streamlit UI
st.title("Healthcare Data Management System")

# Load existing patient data
patient_data = load_patient_data("patient_data.json")

# Add a new patient
st.sidebar.header("Add New Patient")
patient_id = st.sidebar.text_input("Patient ID")
name = st.sidebar.text_input("Name")
medical_records = st.sidebar.text_area("Medical Records")
images = st.sidebar.text_area("Image Paths (comma-separated)")

if st.sidebar.button("Add Patient"):
    images = [img.strip() for img in images.split(",")]
    add_patient(patient_id, name, medical_records, images)
    save_patient_data("patient_data.json")
    st.success("Patient added successfully!")

# Search for patients based on a query
st.header("Search Patients")
query = st.text_input("Search Query")
if st.button("Search"):
    search_results = search_patients(query)
    if search_results:
        st.write("Search results:")
        for patient_id, score in search_results:
            st.write(f"Patient ID: {patient_id}, Similarity Score: {score}")
    else:
        st.warning("No matching patients found.")

# View patient information
st.header("View Patient Information")
view_patient_id = st.text_input("Enter Patient ID to view details")
if st.button("View"):
    patient_info = view_patient(view_patient_id)
    if patient_info:
        st.write("Patient Information:")
        st.write(f"Name: {patient_info['name']}")
        st.write(f"Medical Records: {patient_info['medical_records']}")
        st.write(f"Images: {', '.join(patient_info['images'])}")
    else:
        st.warning("Patient not found.")

# Analyze patient images
st.header("Analyze Patient Images")
if st.button("Analyze Images"):
    if patient_info:
        image_results = analyze_images(patient_info["images"])
        if image_results:
            st.write("Image Analysis Results:")
            for result in image_results:
                st.write(f"Image Path: {result['image_path']}")
                st.write(f"Analysis Result: {result['result']}")
        else:
            st.warning("No images to analyze.")
    else:
        st.warning("Please select a patient to analyze their images.")
