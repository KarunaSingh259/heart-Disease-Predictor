import streamlit as st
import numpy as np
import pandas as pd
import pickle
import onnxruntime as ort

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="💖",
    layout="wide"
)

st.title("💖 Heart Disease Prediction System")

# ---------------- LOAD ML MODELS ----------------
@st.cache_resource
def load_ml_models():
    models = {}
    try:
        models["Decision Tree"] = pickle.load(open("decision_tree_model.pkl", "rb"))
        models["Random Forest"] = pickle.load(open("random_forest_model.pkl", "rb"))
        models["SVM"] = pickle.load(open("svm_model.pkl", "rb"))
        models["Logistic Regression"] = pickle.load(open("LogisticRegressionmodel.pkl", "rb"))
        models["Voting Classifier"] = pickle.load(open("voting_classifier_model.pkl", "rb"))
    except Exception as e:
        st.error(f"❌ Error loading ML models: {e}")
    return models

ml_models = load_ml_models()

# ---------------- LOAD ONNX MODELS ----------------
@st.cache_resource
def load_onnx_model(path):
    try:
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as e:
        st.error(f"❌ Failed to load ONNX model {path}: {e}")
        return None

efficientnet_session = load_onnx_model("efficientnet_ecg_model.onnx")
hybrid_session = load_onnx_model("hybrid_ecg_model.onnx")

# ---------------- TABS ----------------
tabs = st.tabs([
    "🧪 Single Prediction",
    "📊 Batch Prediction",
    "📈 Model Comparison",
    "📁 Dataset Preview",
    "🫀 ECG Deep Learning "
])

# ==================================================
# TAB 1: SINGLE PREDICTION
# ==================================================
with tabs[0]:
    st.subheader("🧪 Single Patient Prediction")

    age = st.number_input("Age", 20, 100, 45)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])

    features = np.array([[age, sex, cp, trestbps, chol, fbs, thalach, exang]])

    model_choice = st.selectbox("Select Model", list(ml_models.keys()))

    if st.button("Predict"):
        model = ml_models[model_choice]
        result = model.predict(features)[0]
        st.success("❤️ Heart Disease Detected" if result == 1 else "✅ No Heart Disease")

# ==================================================
# TAB 2: BATCH PREDICTION
# ==================================================
with tabs[1]:
    st.subheader("📊 Batch Prediction using CSV")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        model_choice = st.selectbox("Select Model", list(ml_models.keys()), key="batch")
        model = ml_models[model_choice]

        if st.button("Predict Batch"):
            preds = model.predict(df.values)
            df["Prediction"] = preds
            st.success("Batch Prediction Completed")
            st.dataframe(df)

# ==================================================
# TAB 3: MODEL COMPARISON
# ==================================================
with tabs[2]:
    st.subheader("📈 Model Comparison")

    sample = np.array([[45, 1, 2, 120, 200, 0, 150, 0]])

    results = {}
    for name, model in ml_models.items():
        results[name] = model.predict(sample)[0]

    st.json(results)

# ==================================================
# TAB 4: DATASET PREVIEW
# ==================================================
with tabs[3]:
    st.subheader("📁 Dataset Preview")

    try:
        df = pd.read_csv("heart (3).csv")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)
    except:
        st.warning("Dataset file not found")

# ==================================================
# TAB 5: ECG DEEP LEARNING (ONNX)
# ==================================================
with tabs[4]:
    st.subheader("🫀 ECG Classification (ONNX Models)")

    st.info("✔ ONNX models work on Streamlit Cloud (No PyTorch needed)")

    uploaded_ecg = st.file_uploader("Upload ECG NumPy File (.npy)", type=["npy"])

    model_type = st.radio(
        "Select ECG Model",
        ["EfficientNet ECG", "Hybrid ECG"]
    )

    if uploaded_ecg:
        ecg_data = np.load(uploaded_ecg)
        ecg_data = ecg_data.astype(np.float32)
        ecg_data = np.expand_dims(ecg_data, axis=0)

        if model_type == "EfficientNet ECG" and efficientnet_session:
            inputs = {efficientnet_session.get_inputs()[0].name: ecg_data}
            output = efficientnet_session.run(None, inputs)
            st.success(f"Prediction Output: {output}")

        elif model_type == "Hybrid ECG" and hybrid_session:
            inputs = {hybrid_session.get_inputs()[0].name: ecg_data}
            output = hybrid_session.run(None, inputs)
            st.success(f"Prediction Output: {output}")

        else:
            st.error("Selected model not available")
