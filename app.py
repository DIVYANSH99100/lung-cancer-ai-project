from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Lung Cancer Classifier", page_icon="🫁", layout="centered")

st.title("🫁 Lung Cancer Detection System")
st.write("Upload a lung CT scan to classify it as **Benign**, **Malignant**, or **Normal**.")

# 2. LOAD THE MULTI-CLASS MODEL
@st.cache_resource
def load_model():
    # Make sure this filename matches your newly trained model
    return tf.keras.models.load_model('lung_cancer_google_v3.h5')

model = load_model()

# 3. UPLOAD INTERFACE
uploaded_file = st.file_uploader("Upload CT Scan (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Scan', use_column_width=True)
    
    with st.spinner('AI is analyzing the scan...'):
        # 4. PREPROCESSING (Must match training: 224x224 and /255)
        # 4. PREPROCESSING (Updated for Google MobileNetV2)
        img = img.convert('RGB') 
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        # WE REMOVED / 255.0 AND ADDED THIS:
        img_array = preprocess_input(img_array)

        # 5. PREDICTION
        prediction = model.predict(img_array)
        
        # Mapping results to classes (Alphabetical order based on folder names)
        # Usually: 0: Benign, 1: Malignant, 2: Normal
        classes = ['Benign', 'Malignant', 'Normal']
        result_index = np.argmax(prediction)
        result_label = classes[result_index]
        confidence = prediction[0][result_index] * 100

    # 6. DISPLAY RESULTS
    st.subheader("Analysis Results:")
    
    # --- ADD THIS TEMP SECTION TO FIX THE MAPPING ---
    st.write("### 🛠 Debugging: AI Index Check")
    st.write(f"The AI is predicting **Index {result_index}**")
    # -----------------------------------------------

    if result_label == 'Malignant':
        st.error(f"**Diagnosis: {result_label}**")
        st.write("The AI has detected patterns associated with cancerous tissue.")
    elif result_label == 'Benign':
        st.warning(f"**Diagnosis: {result_label}**")
        st.write("The AI detected a growth, but it appears to be non-cancerous.")
    else:
        st.success(f"**Diagnosis: {result_label}**")
        st.write("The scan appears healthy with no visible abnormalities.")

    # Visual Confidence Meter
    st.write(f"**AI Confidence Level:** {confidence:.2f}%")
    st.progress(int(confidence))

    # Detailed breakdown for the Project File
    with st.expander("See Raw Probability Breakdown"):
        for i in range(len(classes)):
            st.write(f"{classes[i]}: {prediction[0][i]*100:.2f}%")

st.markdown("---")
st.info("💡 **Project Note:** This model was trained on a specific dataset for CBSE Class 11/12 Computer Science. In a real medical setting, results must be verified by a Radiologist.")