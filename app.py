import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import pytesseract
import cv2
from PIL import Image
import joblib
import pyttsx3
import threading  # Import threading for asynchronous execution

# Load trained ML model
model = joblib.load("model.pkl")  # Ensure you have a trained model

# Disease to doctor mapping
disease_to_doctor = {
    "Diabetes": "Endocrinologist",
    "Heart Disease": "Cardiologist",
    "Lung Disease": "Pulmonologist",
    "Kidney Disease": "Nephrologist",
    "Liver Disease": "Hepatologist",
    "Brain Disorders (Stroke, Epilepsy, etc.)": "Neurologist",
    "Mental Health Disorders": "Psychiatrist",
    "Skin Diseases": "Dermatologist",
    "Eye Diseases": "Ophthalmologist",
    "Ear, Nose, Throat (ENT) Problems": "Otolaryngologist (ENT Specialist)",
    "Bone and Joint Disorders": "Orthopedic Surgeon",
    "Digestive System Disorders (Ulcers, IBS, etc.)": "Gastroenterologist",
    "Reproductive Health Issues (Female)": "Gynecologist",
    "Reproductive Health Issues (Male)": "Andrologist/Urologist",
    "Urinary Tract Issues": "Urologist",
    "Cancer (Various Types)": "Oncologist",
    "Blood Disorders (Anemia, Leukemia, etc.)": "Hematologist",
    "Autoimmune Diseases": "Rheumatologist",
    "Allergic Conditions": "Allergist/Immunologist",
    "Infectious Diseases (HIV, Tuberculosis, etc.)": "Infectious Disease Specialist",
    "Children's Health Issues": "Pediatrician",
    "Pregnancy and Childbirth": "Obstetrician",
    "Hormonal Imbalances": "Endocrinologist",
    "Obesity and Weight Management": "Bariatric Specialist",
    "Dental Issues": "Dentist",
    "Gum and Oral Diseases": "Periodontist",
    "Spinal Cord and Nervous System Issues": "Neurosurgeon",
    "Pain Management": "Pain Specialist/Anesthesiologist",
    "Sleep Disorders": "Sleep Specialist",
    "Emergency Medicine": "Emergency Medicine Physician",
    "General Medical Issues": "General Physician"
}

# Function to extract text from a PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# Function to extract text from an image
def extract_text_from_image(image):
    img = Image.open(image)
    img = np.array(img)
    text = pytesseract.image_to_string(img)
    return text

# Prediction function
def predict_disease(report_text):
    if "glucose" in report_text.lower():
        return "Diabetes"
    elif "chest pain" in report_text.lower():
        return "Heart Disease"
    elif "cough" in report_text.lower():
        return "Lung Disease"
    else:
        return "Unknown Disease"

# Function to generate voice output asynchronously
def speak(text):
    def run_speech():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    
    threading.Thread(target=run_speech, daemon=True).start()

# Streamlit UI
st.title("Medical Diagnosis Prediction System")
st.write("Upload your medical report (PDF or Image) to predict the disease and get doctor recommendations.")

# File uploader
uploaded_file = st.file_uploader("Upload Diagnosis Report", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.write("File uploaded successfully!")

    # Check file type and extract text
    if uploaded_file.type == "application/pdf":
        report_text = extract_text_from_pdf(uploaded_file)
    else:
        report_text = extract_text_from_image(uploaded_file)

    st.subheader("Extracted Report Text:")
    st.write(report_text[:500])  # Show first 500 characters

    # Predict disease
    predicted_disease = predict_disease(report_text)
    st.subheader(f"Predicted Disease: {predicted_disease}")

    # Suggest a doctor
    doctor = disease_to_doctor.get(predicted_disease, "General Physician")
    st.subheader(f"Suggested Doctor: {doctor}")

    # Generate voice output
    result_text = f"The predicted disease is {predicted_disease}. You should consult a {doctor}."
    speak(result_text)
    st.write("🔊 **Voice Output Generated**")
