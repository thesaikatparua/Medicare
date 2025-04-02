import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import pytesseract
import cv2
from PIL import Image
import joblib
import pyttsx3
import threading
import time
from fpdf import FPDF

# Load trained ML model
model = joblib.load("model.pkl")  # Ensure you have a trained model

# Disease to doctor mapping
disease_to_doctor = {
    "Diabetes": "Endocrinologist",
    "Heart Disease": "Cardiologist",
    "Lung Disease": "Pulmonologist",
    "Kidney Disease": "Nephrologist",
    "Liver Disease": "Hepatologist",
    "Stroke": "Neurologist",
    "Epilepsy": "Neurologist",
    "Parkinson's Disease": "Neurologist",
    "Arthritis": "Rheumatologist",
    "Osteoporosis": "Orthopedic Surgeon",
    "Psoriasis": "Dermatologist",
    "Eczema": "Dermatologist",
    "Acne": "Dermatologist",
    "Cataracts": "Ophthalmologist",
    "Glaucoma": "Ophthalmologist",
    "Sinusitis": "Otolaryngologist",
    "Hearing Loss": "Otolaryngologist",
    "Ulcer": "Gastroenterologist",
    "IBS": "Gastroenterologist",
    "Crohn’s Disease": "Gastroenterologist",
    "Anemia": "Hematologist",
    "Leukemia": "Hematologist/Oncologist",
    "Thyroid Issues": "Endocrinologist",
    "PCOS": "Endocrinologist/Gynecologist",
    "Kidney Stones": "Urologist",
    "UTI": "Urologist",
    "Breast Cancer": "Oncologist",
    "Lung Cancer": "Oncologist",
    "Depression": "Psychiatrist",
    "Anxiety": "Psychiatrist",
    "Schizophrenia": "Psychiatrist",
    "Tuberculosis": "Infectious Disease Specialist",
    "HIV/AIDS": "Infectious Disease Specialist",
    "COVID-19": "Infectious Disease Specialist",
    "Asthma": "Allergist/Immunologist",
    "Hay Fever": "Allergist/Immunologist",
    "Pregnancy": "Gynecologist/Obstetrician",
    "ADHD": "Pediatrician",
    "Childhood Asthma": "Pediatrician"
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
>>>>>>> ab35e72737fc6a0738b78d637f81c371dfa6eb6e
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
    predictions = {
        "Diabetes": 0.8 if "glucose" in report_text.lower() else 0.2,
        "Heart Disease": 0.7 if "chest pain" in report_text.lower() else 0.3,
        "Lung Disease": 0.6 if "cough" in report_text.lower() else 0.4,
    }
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return sorted_predictions

# Function to generate voice output asynchronously
def speak(text):
    def run_speech():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    
    threading.Thread(target=run_speech, daemon=True).start()

def get_google_maps_link(doctor):
    base_url = "https://www.google.com/maps/search/"
    search_query = f"{doctor} near me"
    return f"[Find a {doctor} Near You]({base_url}{search_query})"

# Streamlit UI
st.title("Medicare – Your Smart Health Companion !")
st.write("Upload your medical report (PDF or Image) to predict the disease and get doctor recommendations.")

# File uploader
uploaded_file = st.file_uploader("Upload Diagnosis Report", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.write("File uploaded successfully!")
    st.write("Processing the file...")
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.05)  # Simulating processing time
        progress_bar.progress(i + 1)

    # Check file type and extract text
    if uploaded_file.type == "application/pdf":
        report_text = extract_text_from_pdf(uploaded_file)
    else:
        report_text = extract_text_from_image(uploaded_file)

    with st.expander("Extracted Report Text", expanded=True):
        edited_text = st.text_area("Edit the extracted text before prediction", report_text, height=200)

    # Predict disease
    predictions = predict_disease(edited_text)
    disease_names, confidence_scores = zip(*predictions)

    # Display bar chart for confidence scores
    st.subheader("Predicted Diseases with Confidence Levels")
    st.bar_chart(pd.DataFrame({"Confidence": confidence_scores}, index=disease_names))

    # Get the top predicted disease
    top_disease = disease_names[0]  # Extract top disease name
    doctor = disease_to_doctor.get(top_disease, "General Physician")

    st.subheader(f"Suggested Doctor: {doctor}")
    st.markdown(get_google_maps_link(doctor))

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.subheader("Ask Our Virtual Health Assistant")
    user_input = st.text_input("Type your question:")

    if st.button("Ask"):
        responses = {
            "what is diabetes": "Diabetes is a condition that affects how your body processes blood sugar.",
            "what is heart disease": "Heart disease refers to conditions affecting the heart, like coronary artery disease.",
        }
        response = responses.get(user_input.lower(), "Sorry, I don't have an answer for that.")
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    for sender, message in st.session_state.chat_history:
        st.write(f"**{sender}:** {message}")

    # Generate voice output
    result_text = f"The predicted disease is {top_disease}. You should consult a {doctor}."
    speak(result_text)
    st.write("🔊 **Voice Output Generated**")

    # Enable voice settings
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = True
    if "voice_speed" not in st.session_state:
        st.session_state.voice_speed = 150

    voice_enabled = st.checkbox("Enable Voice Output", value=st.session_state.voice_enabled)
    voice_speed = st.slider("Voice Speed", min_value=100, max_value=300, value=st.session_state.voice_speed)

    def speak(text):
        if voice_enabled:
            engine = pyttsx3.init()
            engine.setProperty("rate", voice_speed)
            engine.say(text)
            engine.runAndWait()

    # PDF Report Generation
    def generate_pdf(disease, doctor):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Medical Diagnosis Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Predicted Disease: {disease}", ln=True)
        pdf.cell(200, 10, txt=f"Suggested Doctor: {doctor}", ln=True)
        
        pdf_filename = "diagnosis_report.pdf"
        pdf.output(pdf_filename)
        return pdf_filename

    if st.button("Download Report"):
        pdf_file = generate_pdf(top_disease, doctor)
        with open(pdf_file, "rb") as file:
            st.download_button(label="Download Report", data=file, file_name=pdf_file)



