import streamlit as st
import numpy as np
import pandas as pd
import pickle
import onnxruntime as ort
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD LOGO ----------------
st.sidebar.image("logo.png", width=200)
st.sidebar.title("❤️ HEART DISEASE PREDICTOR")
st.sidebar.write("Your intelligent cardiac care assistant")

# ---------------- LOAD ML MODELS ----------------
@st.cache_resource
def load_ml_models():
    models = {
        "Voting": pickle.load(open("voting_classifier_model.pkl", "rb")),
        "Random Forest": pickle.load(open("random_forest_model.pkl", "rb")),
        "SVM": pickle.load(open("svm_model.pkl", "rb")),
        "Logistic Regression": pickle.load(open("LogisticRegressionmodel.pkl", "rb")),
        "Decision Tree": pickle.load(open("decision_tree_model.pkl", "rb")),
    }
    return models

ml_models = load_ml_models()

# ---------------- LOAD ONNX MODELS ----------------
@st.cache_resource
def load_onnx_models():
    eff_sess = ort.InferenceSession(
        "models/efficientnet_ecg_model.onnx",
        providers=["CPUExecutionProvider"]
    )
    hybrid_sess = ort.InferenceSession(
        "models/hybrid_ecg_model.onnx",
        providers=["CPUExecutionProvider"]
    )
    return eff_sess, hybrid_sess

eff_session, hybrid_session = load_onnx_models()

# ---------------- TABS ----------------
tabs = st.tabs([
    "🔍 Predict",
    "📊 Prediction Result",
    "📁 Bulk Predict",
    "📘 Model Info",
    "🫀 ECG Prediction"
])

# ---------------- TAB 1: PREDICT ----------------
with tabs[0]:
    st.header("🔍 Heart Disease Prediction")

    age = st.number_input("Age", 1, 120, 45)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    if st.button("Predict"):
        results = {}
        for name, model in ml_models.items():
            results[name] = model.predict(input_data)[0]

        st.session_state["results"] = results
        st.success("Prediction completed! Go to Prediction Result tab.")

# ---------------- TAB 2: RESULT ----------------
with tabs[1]:
    st.header("📊 Prediction Result")

    if "results" in st.session_state:
        for model, res in st.session_state["results"].items():
            st.write(f"**{model}** → {'Heart Disease' if res == 1 else 'No Disease'}")
    else:
        st.info("Run prediction first.")

# ---------------- TAB 3: BULK ----------------
with tabs[2]:
    st.header("📁 Bulk Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        model = ml_models["Voting"]
        df["Prediction"] = model.predict(df)
        st.dataframe(df)

# ---------------- TAB 4: MODEL INFO ----------------
with tabs[3]:
    st.header("📘 Model Information")
    st.markdown("""
    **Models Used:**
    - Logistic Regression
    - Support Vector Machine
    - Random Forest
    - Decision Tree
    - Voting Classifier

    **ECG Models (ONNX):**
    - EfficientNet-B0
    - Hybrid ResNet-18
    """)

# ---------------- TAB 5: ECG PREDICTION (ONNX) ----------------
with tabs[4]:
    st.header("🫀 ECG Disease Prediction (Deep Learning)")

    uploaded_img = st.file_uploader(
        "Upload ECG Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="Uploaded ECG", width=300)

        img = img.resize((224, 224))
        img = np.array(img).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        eff_out = eff_session.run(
            None,
            {"input": img}
        )[0]

        hyb_out = hybrid_session.run(
            None,
            {"input": img}
        )[0]

        eff_pred = np.argmax(eff_out)
        hyb_pred = np.argmax(hyb_out)

        st.subheader("Results")
        st.write(f"**EfficientNet Prediction:** Class {eff_pred}")
        st.write(f"**Hybrid Model Prediction:** Class {hyb_pred}")

        final_pred = round((eff_pred + hyb_pred) / 2)
        st.success(f"🫀 Final ECG Prediction Class: {final_pred}")
