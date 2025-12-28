import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import onnxruntime as ort
from PIL import Image

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction System")

# ==============================
# LOAD ML MODELS (SAFE)
# ==============================
@st.cache_resource
def load_ml_models():
    models = {
        "Decision Tree": pickle.load(open("decision_tree_model.pkl", "rb")),
        "Logistic Regression": pickle.load(open("LogisticRegressionmodel.pkl", "rb")),
        "Random Forest": pickle.load(open("random_forest_model.pkl", "rb")),
        "SVM": pickle.load(open("svm_model.pkl", "rb"))
    }
    return models

ml_models = load_ml_models()

# ==============================
# LOAD ONNX ECG MODELS
# ==============================
@st.cache_resource
def load_onnx_models():
    eff_sess = ort.InferenceSession("efficientnet_ecg_model.onnx")
    hyb_sess = ort.InferenceSession("hybrid_ecg_model.onnx")
    return eff_sess, hyb_sess

eff_sess, hyb_sess = load_onnx_models()

# ==============================
# TABS
# ==============================
tabs = st.tabs([
    "🔮 Predict",
    "📊 Result",
    "📈 Model Comparison",
    "📄 Model Info",
    "🫀 ECG Image Test"
])

# ==============================
# TAB 1 – INPUT
# ==============================
with tabs[0]:
    st.subheader("🔮 Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting BP", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        restecg = st.selectbox("Rest ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Angina", [0, 1])
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope", [0, 1, 2])
       

    sex = 1 if sex == "Male" else 0

    sample = np.array([[age, sex, cp, trestbps, chol, fbs,
                         restecg, thalach, exang, oldpeak,
                         slope,]])

    if st.button("Predict"):
        st.session_state["sample"] = sample
        st.success("Data saved. Go to Result tab.")

# ==============================
# TAB 2 – RESULT
# ==============================
with tabs[1]:
    st.subheader("📊 Prediction Result")

    if "sample" not in st.session_state:
        st.warning("Please predict first.")
    else:
        sample = st.session_state["sample"]

        preds = []
        for model in ml_models.values():
            preds.append(model.predict(sample)[0])

        risk = np.mean(preds) * 100

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={"text": "Heart Disease Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 40], "color": "lightgreen"},
                    {"range": [40, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"}
                ]
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

# ==============================
# TAB 3 – MODEL COMPARISON (FIXED)
# ==============================
with tabs[2]:
    st.subheader("📈 Model Comparison")

    sample = np.array([[45,1,2,120,200,0,1,150,0,2.3,1,0,2]])

    results = {}
    for name, model in ml_models.items():
        results[name] = int(model.predict(sample)[0])

    st.json(results)

# ==============================
# TAB 4 – INFO
# ==============================
with tabs[3]:
    st.subheader("📄 Model Accuracy")

    acc = {
        "Decision Tree": 80.9,
        "Logistic Regression": 85.8,
        "Random Forest": 84.2,
        "SVM": 84.2
    }

    df = pd.DataFrame(acc.items(), columns=["Model", "Accuracy"])
    st.bar_chart(df.set_index("Model"))

# ==============================
# TAB 5 – ECG IMAGE (ONNX)
# ==============================
with tabs[4]:
    st.subheader("🫀 ECG Image Diagnosis (ONNX)")

    uploaded = st.file_uploader("Upload ECG Image", ["jpg", "png"])

    labels = ["Normal", "Myocardial Infarction", "Abnormal Heartbeat", "History of MI"]

    if uploaded:
        img = Image.open(uploaded).convert("RGB").resize((224, 224))
        st.image(img, caption="Uploaded ECG")

        img_arr = np.array(img).astype(np.float32) / 255.0
        img_arr = np.transpose(img_arr, (2, 0, 1))
        img_arr = np.expand_dims(img_arr, axis=0)

        eff_pred = eff_sess.run(None, {"input": img_arr})[0]
        hyb_pred = hyb_sess.run(None, {"input": img_arr})[0]

        eff_class = labels[np.argmax(eff_pred)]
        hyb_class = labels[np.argmax(hyb_pred)]

        st.success(f"EfficientNet: {eff_class}")
        st.success(f"Hybrid Model: {hyb_class}")


