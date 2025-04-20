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

# ✅ Streamlit Page Configuration
st.set_page_config(
    page_title="Medicare – Smart Health Companion",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="auto"
)

# ✅ Background Style
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1585435557343-3b092031a831?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
            background-size: cover;
        }
        .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
set_background()

# ✅ Sidebar Navigation
st.sidebar.title("🧭 Medicare")
page = st.sidebar.radio("Go to", ["🏠 Home", "📁 Upload Report", "🗣️ Chat Assistant", "📜 Download Report"])

# ✅ Title and Subtitle
st.markdown('<div style="text-align: center;"><h1 style="color:#fff;">Medicare – Your Smart Health Companion 🩺</h1></div>', unsafe_allow_html=True)
st.markdown('<h4 style="color:white; text-align:center;">📄 Upload Repor |👨‍⚕️ Get Doctor</h4>', unsafe_allow_html=True)

# ✅ Disease to Doctor Mapping
disease_to_doctor = {
    "Diabetes": "Endocrinologist",
    "Heart Disease": "Cardiologist",
    "Lung Disease": "Pulmonologist",
}

# ✅ Text Extraction Functions
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_image(image):
    img = Image.open(image)
    img = np.array(img)
    return pytesseract.image_to_string(img)

# ✅ Prediction Logic (Sample Rule-Based)
def predict_disease(report_text):
    predictions = {
        "Diabetes": 0.8 if "glucose" in report_text.lower() else 0.2,
        "Heart Disease": 0.7 if "chest pain" in report_text.lower() else 0.3,
        "Lung Disease": 0.6 if "cough" in report_text.lower() else 0.4,
    }
    return sorted(predictions.items(), key=lambda x: x[1], reverse=True)

# ✅ Asynchronous Voice Function
def speak_async(text, speed=150):
    def run():
        engine = pyttsx3.init()
        engine.setProperty('rate', speed)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

# ✅ Google Maps Doctor Link
def get_google_maps_link(doctor):
    return f"[Find a {doctor} Near You](https://www.google.com/maps/search/{doctor}+near+me)"

# ✅ Upload Page
if page == "🏠 Home":
    st.markdown('<h2 style="color:white;">📄 Upload Diagnosis Report</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a diagnosis report", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file:
        st.success("✅ File uploaded! Processing...")

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = extract_text_from_image(uploaded_file)

        st.markdown('<h3 style="color:white;">📝 Extracted Text</h3>', unsafe_allow_html=True)
        edited_text = st.text_area("You can edit the text below before prediction:", text, height=200)

        predictions = predict_disease(edited_text)
        diseases, scores = zip(*predictions)

        st.markdown('<h3 style="color:white;">📊 Prediction Confidence</h3>', unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame({"Confidence": scores}, index=diseases))

        top_disease = diseases[0]
        doctor = disease_to_doctor.get(top_disease, "General Physician")
        st.markdown(f'<h3 style="color:white;">👨‍⚕️ Suggested Doctor: {doctor}</h3>', unsafe_allow_html=True)
        st.markdown(get_google_maps_link(doctor))

        voice_enabled = st.checkbox("🔊 Enable Voice Output", value=True)
        voice_speed = st.slider("Voice Speed", 100, 300, 150)

        if voice_enabled:
            speak_async(f"The predicted disease is {top_disease}. Please consult a {doctor}.", speed=voice_speed)
            

# ✅ Chat Assistant Page
elif page == "🗣️ Chat Assistant":
    st.markdown('<h2 style="color:white;">🤖 Chat with Assistant</h2>', unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask about a disease:")
    if st.button("Ask"):
        responses = {
    "what is diabetes": "Diabetes is a condition where blood glucose is too high. It occurs when the body either doesn't produce enough insulin or can't use insulin properly.",
    
    "what is heart disease": "Heart disease refers to various conditions that affect the heart, including coronary artery disease, heart attack, and heart failure.",
    
    "what is hypertension": "Hypertension, or high blood pressure, is a condition where the force of the blood against the walls of your arteries is too high.",
    
    "what is cancer": "Cancer is a group of diseases involving abnormal cell growth with the potential to spread to other parts of the body.",
    
    "what is stroke": "A stroke occurs when there is a blockage or rupture of blood vessels in the brain, leading to brain tissue damage.",
    
    "what is asthma": "Asthma is a condition that causes the airways to become inflamed, leading to difficulty in breathing, wheezing, and coughing.",
    
    "what is pneumonia": "Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus, making it difficult to breathe.",
    
    "what is tuberculosis": "Tuberculosis (TB) is a bacterial infection that primarily affects the lungs but can also affect other parts of the body.",
    
    "what is arthritis": "Arthritis is an inflammation of one or more joints, causing pain and stiffness that can worsen with age.",
    
    "what is migraine": "A migraine is a severe, throbbing headache often accompanied by nausea, vomiting, and sensitivity to light and sound.",
    
    "what is Alzheimer’s disease": "Alzheimer's disease is a progressive neurological disorder that causes memory loss, confusion, and changes in behavior.",
    
    "what is depression": "Depression is a mood disorder that causes persistent feelings of sadness, loss of interest, and other emotional and physical symptoms.",
    
    "what is anxiety": "Anxiety is a feeling of worry, nervousness, or unease, often about an imminent event or something with an uncertain outcome.",
    
    "what is osteoporosis": "Osteoporosis is a condition where bones become weak and brittle, increasing the risk of fractures.",
    
    "what is epilepsy": "Epilepsy is a neurological disorder marked by recurrent seizures due to abnormal electrical activity in the brain.",
    
    "what is HIV/AIDS": "HIV is a virus that attacks the immune system, and if left untreated, it can lead to AIDS, a condition where the immune system is severely weakened.",
    
    "what is kidney disease": "Kidney disease refers to conditions that impair kidney function, including chronic kidney disease, kidney stones, and infections.",
    
    "what is cirrhosis": "Cirrhosis is scarring of the liver caused by long-term liver damage, often from alcohol use, hepatitis, or fatty liver disease.",
    
    "what is eczema": "Eczema is a condition that causes the skin to become inflamed, itchy, red, and cracked, often triggered by allergens or irritants.",
    
    "what is psoriasis": "Psoriasis is a chronic skin condition that causes rapid skin cell turnover, resulting in patches of red, scaly skin.",
    
    "what is a cold": "A cold is a viral infection that affects the upper respiratory tract, causing symptoms like a runny nose, sore throat, and coughing.",
    
    "what is flu": "The flu, or influenza, is a contagious viral infection that affects the respiratory system, causing fever, body aches, fatigue, and cough.",
    
    "what is vertigo": "Vertigo is a sensation of dizziness or spinning, often caused by issues with the inner ear or balance system.",
    
    "what is hepatitis": "Hepatitis is an inflammation of the liver, often caused by viral infections (Hepatitis A, B, C) or alcohol use.",
    
    "what is a panic disorder": "Panic disorder is a mental health condition characterized by recurrent and unexpected panic attacks.",
    
    "what is a urinary tract infection (UTI)": "A UTI is an infection in any part of the urinary system, commonly caused by bacteria entering the urethra and bladder.",
    
    "how to connect with a doctor": "You can connect with a doctor through telemedicine services, by booking an appointment at a nearby clinic, or using health apps that offer consultations. You can also search for doctors in your area through Google Maps.",
    
    "how to book a doctor appointment": "To book a doctor's appointment, you can either call the clinic directly, use an online appointment booking system through healthcare apps, or visit the clinic's website for more details.",
    
    "how to prepare for a doctor's visit": "It's important to write down your symptoms, medical history, medications, and any questions you may have. Bring a list of current medications and any test results you have received.",
    
    "how to choose a specialist": "Choosing a specialist depends on your symptoms. For example, if you have heart problems, you should see a cardiologist. You can ask your primary care doctor for recommendations or search online.",
    
    "how to know if I need to see a doctor": "If you experience persistent or worsening symptoms, pain, or discomfort, or if you're unsure about your health, it's best to consult a doctor for a professional diagnosis.",
    
    "how can I get my medical records": "You can request your medical records from the hospital or clinic where you received treatment. Some healthcare providers allow you to access them through their patient portal.",
    
    "how to manage stress": "Stress can be managed through regular physical activity, practicing relaxation techniques like meditation and deep breathing, maintaining a balanced diet, and seeking support from loved ones or a counselor.",
    
    "what to do in case of an emergency": "In an emergency, immediately call emergency services (such as 911 or your country's equivalent). If possible, provide clear details about the situation and follow the instructions of the dispatcher.",
    
    "how to check blood pressure at home": "To check your blood pressure at home, you can use a blood pressure monitor. Follow the device's instructions and make sure you're seated comfortably with your arm at heart level.",
    
    "how to prevent heart disease": "To prevent heart disease, maintain a healthy diet, exercise regularly, manage stress, avoid smoking, and monitor your cholesterol and blood pressure levels.",
    
    "how to control blood sugar levels": "To control blood sugar, eat a balanced diet rich in whole grains, vegetables, and lean proteins, exercise regularly, and monitor your blood sugar levels as advised by your doctor.",
    
    "how to lose weight healthily": "To lose weight healthily, focus on a balanced diet, exercise regularly, stay hydrated, and consult a healthcare provider for personalized advice based on your health needs.",
    
    "what is the treatment for anxiety": "Treatment for anxiety can include therapy (like CBT), medication, relaxation techniques, and lifestyle changes such as exercise and improved sleep.",
    
    "what to do if I have a fever": "If you have a fever, drink plenty of fluids, rest, and monitor your temperature. If the fever persists or is very high, consult a healthcare provider.",
    
    "what is a routine check-up": "A routine check-up is a regular health examination where a doctor evaluates your overall health, conducts screenings, and provides preventive care based on your age and medical history.",
    
    "how to prevent infections": "To prevent infections, wash your hands regularly, avoid close contact with sick people, get vaccinated, and practice safe food handling and hygiene.",
    
    "what is the best way to boost immunity": "To boost your immunity, eat a balanced diet with fruits and vegetables, exercise regularly, get enough sleep, manage stress, and avoid smoking.",
    
    "how to know if I have a chronic illness": "Chronic illnesses often have long-lasting symptoms. If you experience symptoms that persist over time, or if you have risk factors such as a family history of disease, consult your doctor for a proper diagnosis.",
    
    "what is a second opinion": "A second opinion is when you seek advice from another doctor, usually after receiving a diagnosis or treatment plan, to ensure you are making the best-informed decision for your health."
}

        answer = responses.get(user_input.lower(), "I'm sorry, I don't have an answer for that yet.")
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", answer))

    for sender, msg in st.session_state.chat_history:
        st.write(f"**{sender}:** {msg}")

# ✅ Download Report Page
elif page == "📜 Download Report":
    st.markdown('<h2 style="color:white;">📥 Download Your Report</h2>', unsafe_allow_html=True)
    def generate_pdf(disease, doctor):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Medical Diagnosis Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Predicted Disease: {disease}", ln=True)
        pdf.cell(200, 10, txt=f"Suggested Doctor: {doctor}", ln=True)
        filename = "diagnosis_report.pdf"
        pdf.output(filename)
        return filename

    top_disease = "Diabetes"  # Placeholder: You can link with session state
    doctor = disease_to_doctor.get(top_disease, "General Physician")

    if st.button("📥 Generate & Download PDF Report"):
        pdf_path = generate_pdf(top_disease, doctor)
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report", f, file_name=pdf_path, mime='application/octet-stream')

# ✅ Home Page
else:
    st.markdown('<h2 style="color:white;">Welcome to Medicare!</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:white;">Use the navigation menu on the left to upload reports, chat with our assistant, or download your medical report.</p>', unsafe_allow_html=True)