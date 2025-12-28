import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Heart Disease App", page_icon="💖", layout="wide")

# ---------------- SAFE PICKLE LOADER ----------------
@st.cache_resource
def load_pickle_model(path):
    if not os.path.exists(path):
        st.error(f"❌ Model file not found: {path}")
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load {path}: {e}")
        return None

# ---------------- DARK MODE ----------------
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

if dark_mode:
    bg_color = "#000000"
    font_color = "white"
    card_bg = "rgba(255,255,255,0.07)"
else:
    bg_color = "#ffffff"
    font_color = "black"
    card_bg = "rgba(255,255,255,0.55)"

# ---------------- CSS ----------------
st.markdown(f"""
<style>
body {{background-color:{bg_color}; color:{font_color};}}
.card {{
    background:{card_bg};
    padding:25px;
    border-radius:18px;
    box-shadow:0px 4px 18px rgba(0,0,0,0.2);
}}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("💖 HEART DISEASE PREDICTOR")
st.sidebar.markdown("Your intelligent cardiac care assistant 💓")
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.header("Built with ML & Deep Health Analytics")

# ---------------- TABS ----------------
st.title("💖 Heart Disease Prediction System")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predict", "📊 Prediction Result", "📂 Bulk Predict",
    "📈 Model Info", "🫀 ECG Model"
])

# ================= TAB 1 =================
with tab1:
    st.header("🔍 Single Prediction")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 30)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type",
                          ["Typical Angina", "Atypical Angina",
                           "Non-Anginal Pain", "Asymptomatic"])
        rbp = st.number_input("Resting BP", 0, 300, 120)
        chol = st.number_input("Cholesterol", 0, 600)
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar", ["<=120", ">120"])
        ecg = st.selectbox("Resting ECG",
                           ["Normal", "ST-T Abnormality", "LVH"])
        maxhr = st.number_input("Max Heart Rate", 60, 202, 150)
        angina = st.selectbox("Exercise Angina", ["Yes", "No"])
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
        slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    input_df = pd.DataFrame({
        "Age": [age],
        "Sex": [0 if sex == "Male" else 1],
        "ChestPainType": [["Typical Angina","Atypical Angina",
                           "Non-Anginal Pain","Asymptomatic"].index(cp)],
        "RestingBP": [rbp],
        "Cholesterol": [chol],
        "FastingBS": [1 if fbs == ">120" else 0],
        "RestingECG": [["Normal","ST-T Abnormality","LVH"].index(ecg)],
        "MaxHR": [maxhr],
        "ExerciseAngina": [1 if angina == "Yes" else 0],
        "Oldpeak": [oldpeak],
        "ST_Slope": [["Up","Flat","Down"].index(slope)]
    })

    if st.button("🔮 Predict Now"):
        st.session_state["data"] = input_df
        st.session_state["show"] = True
        st.success("Data saved. Go to Prediction Result tab.")

# ================= TAB 2 =================
with tab2:
    st.header("📊 Prediction Result")

    if "show" not in st.session_state:
        st.info("Predict first.")
        st.stop()

    models = {
        "Decision Tree": "decision_tree_model.pkl",
        "Logistic Regression": "LogisticRegressionmodel.pkl",
        "Random Forest": "random_forest_model.pkl",
        "SVM": "svm_model.pkl",
        "Voting": "voting_classifier_model.pkl"
    }

    preds = []
    for name, file in models.items():
        model = load_pickle_model(file)
        if model:
            preds.append(model.predict(st.session_state["data"])[0])

    if not preds:
        st.error("No predictions available.")
        st.stop()

    risk = np.mean(preds) * 100

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={"text": "Heart Disease Risk (%)"},
        gauge={"axis":{"range":[0,100]}}
    ))
    st.plotly_chart(gauge, use_column_width=True)

# ================= TAB 3 =================
with tab3:
    st.header("📂 Bulk Prediction")
    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df = pd.read_csv(file)
        if "HeartDisease" in df.columns:
            df = df.drop("HeartDisease", axis=1)

        model = load_pickle_model("LogisticRegressionmodel.pkl")
        if model:
            df["Prediction"] = model.predict(df)
            df["Prediction"] = df["Prediction"].map(
                {0: "No Disease", 1: "Disease"}
            )
            st.write(df)

# ================= TAB 4 =================
with tab4:
    st.header("📈 Model Accuracy")
    acc = {
        "Decision Tree": 80.9,
        "Logistic Regression": 85.8,
        "Random Forest": 84.2,
        "SVM": 84.2,
        "Voting": 89.7
    }
    df = pd.DataFrame(acc.items(), columns=["Model", "Accuracy"])
    st.plotly_chart(px.bar(df, x="Model", y="Accuracy"))

# ================= TAB 5 =================
with tab5:
    st.header("🫀 ECG Diagnosis")

    @st.cache_resource
    def load_ecg_models():
        device = torch.device("cpu")
        eff = models.efficientnet_b0(weights=None)
        eff.classifier[1] = nn.Linear(eff.classifier[1].in_features, 4)
        eff.load_state_dict(torch.load("efficientnet_ecg_model.pth",
                                       map_location=device),
                            strict=False)
        eff.eval()

        hyb = models.resnet18(weights=None)
        hyb.fc = nn.Linear(hyb.fc.in_features, 4)
        hyb.load_state_dict(torch.load("hybrid_ecg_model.pth",
                                       map_location=device),
                            strict=False)
        hyb.eval()
        return eff, hyb

    try:
        eff_model, hyb_model = load_ecg_models()
        st.success("ECG Models Loaded")
    except:
        st.error("ECG models not loaded")
        st.stop()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    img = st.file_uploader("Upload ECG Image", type=["jpg","png"])
    if img:
        image = Image.open(img).convert("RGB")
        st.image(image)
        x = transform(image).unsqueeze(0)
        with torch.no_grad():
            p1 = torch.argmax(eff_model(x),1).item()
            p2 = torch.argmax(hyb_model(x),1).item()
        st.success(f"EfficientNet: {p1} | Hybrid: {p2}")
