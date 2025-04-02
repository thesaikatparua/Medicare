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
    "General Medical Issues": "General Physician",
    "High Blood Pressure (Hypertension)": "Cardiologist",
    "Low Blood Pressure (Hypotension)": "Cardiologist",
    "Chronic Fatigue Syndrome": "Internal Medicine Specialist",
    "Arthritis (Osteoarthritis, Rheumatoid)": "Rheumatologist",
    "Thyroid Disorders (Hyperthyroidism, Hypothyroidism)": "Endocrinologist",
    "Migraines and Chronic Headaches": "Neurologist",
    "Parkinson’s Disease": "Neurologist",
    "Alzheimer’s Disease and Dementia": "Neurologist",
    "Multiple Sclerosis": "Neurologist",
    "Chronic Pain Conditions (Fibromyalgia, etc.)": "Pain Management Specialist",
    "Acid Reflux (GERD)": "Gastroenterologist",
    "Hepatitis": "Hepatologist",
    "Gallbladder Disease (Gallstones, etc.)": "Gastroenterologist",
    "Pancreatitis": "Gastroenterologist",
    "Celiac Disease": "Gastroenterologist",
    "Irritable Bowel Syndrome (IBS)": "Gastroenterologist",
    "Crohn’s Disease and Ulcerative Colitis": "Gastroenterologist",
    "Hernia": "General Surgeon",
    "Hemorrhoids": "Proctologist",
    "Lymphatic Disorders (Lymphedema, etc.)": "Vascular Specialist",
    "Varicose Veins": "Vascular Surgeon",
    "Chronic Obstructive Pulmonary Disease (COPD)": "Pulmonologist",
    "Asthma": "Pulmonologist",
    "Pneumonia": "Pulmonologist",
    "Tuberculosis": "Infectious Disease Specialist",
    "Sinusitis and Nasal Polyps": "ENT Specialist",
    "Tonsillitis": "ENT Specialist",
    "Hearing Loss and Ear Infections": "ENT Specialist",
    "Vertigo and Balance Disorders": "ENT Specialist",
    "Speech and Swallowing Disorders": "Speech Therapist",
    "Autism Spectrum Disorders": "Developmental Pediatrician",
    "Attention Deficit Hyperactivity Disorder (ADHD)": "Psychiatrist",
    "Depression and Anxiety": "Psychiatrist",
    "Bipolar Disorder": "Psychiatrist",
    "Schizophrenia": "Psychiatrist",
    "Post-Traumatic Stress Disorder (PTSD)": "Psychiatrist",
    "Obsessive-Compulsive Disorder (OCD)": "Psychiatrist",
    "Menstrual Disorders": "Gynecologist",
    "Polycystic Ovary Syndrome (PCOS)": "Gynecologist",
    "Infertility": "Reproductive Endocrinologist",
    "Menopause-Related Issues": "Gynecologist",
    "Sexually Transmitted Diseases (STDs)": "Infectious Disease Specialist",
    "Prostate Disorders": "Urologist",
    "Erectile Dysfunction": "Andrologist",
    "Hair Loss and Scalp Disorders": "Dermatologist",
    "Acne and Skin Infections": "Dermatologist",
    "Psoriasis and Eczema": "Dermatologist",
    "Skin Cancer": "Dermatologist/Oncologist",
    "Vitiligo": "Dermatologist",
    "Fractures and Sports Injuries": "Orthopedic Surgeon",
    "Osteoporosis": "Endocrinologist/Rheumatologist",
    "Foot and Ankle Disorders": "Podiatrist",
    "Hand and Wrist Disorders": "Orthopedic Surgeon",
    "Burns and Severe Wounds": "Plastic Surgeon",
    "Plastic and Reconstructive Surgery": "Plastic Surgeon",
    "Sleep Apnea": "Sleep Specialist",
    "Chronic Insomnia": "Sleep Specialist",
    "Eating Disorders (Anorexia, Bulimia)": "Psychiatrist",
    "Tobacco, Alcohol, or Drug Addiction": "Addiction Specialist",
    "Work-Related Stress Disorders": "Psychologist",
    "Geriatric Health Issues (Elderly Care)": "Geriatrician",
    "Occupational Health Disorders": "Occupational Medicine Specialist",
    "Genetic Disorders (Down Syndrome, Cystic Fibrosis, etc.)": "Geneticist",
    "Vaccination and Immunization": "General Physician/Pediatrician",
    "Travel Medicine (Malaria, Yellow Fever, etc.)": "Travel Medicine Specialist",
    "Rare and Undiagnosed Conditions": "Internal Medicine Specialist"
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
