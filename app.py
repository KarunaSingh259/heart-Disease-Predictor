import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from torchvision import transforms
import onnxruntime as ort

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(page_title="Heart Disease App", page_icon="💖", layout="wide")

# --------------------- DARK MODE TOGGLE ---------------------
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")
if dark_mode:
    bg_color = "#000000"
    font_color = "white"
    card_bg = "rgba(255,255,255,0.07)"
else:
    bg_color = "#ffffff"
    font_color = "black"
    card_bg = "rgba(255,255,255,0.55)"

# --------------------- CUSTOM CSS ---------------------
st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {font_color};
}}
.card {{
    background: {card_bg};
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.2);
    margin-bottom: 20px;
}}
.stButton>button {{
    background: linear-gradient(to right,#ff4b6e,#ff0055);
    color: white;
    border-radius: 12px;
    padding: 10px 22px;
    border: none;
    font-size: 18px;
    transition: 0.3s;
}}
.stButton>button:hover {{
    transform: scale(1.03);
}}
[data-testid="stSidebar"] {{
    background: {card_bg};
    backdrop-filter: blur(8px);
}}
</style>
""", unsafe_allow_html=True)

# --------------------- SIDEBAR ---------------------
st.sidebar.title("💖 HEART DISEASE PREDICTOR")
st.sidebar.markdown("Your intelligent cardiac care assistant 💓")
st.sidebar.header("Built with ML & Deep Health Analytics")

# ================================================================
# ✅ TABS
# ================================================================
st.title("💖 Heart Disease Prediction System")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predict", "📊 Prediction Result", "📂 Bulk Predict", "📈 Model Info", "🫀 ECG Model"
])

# ================================================================
# TAB 1 — PREDICT
# ================================================================
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🔍 Single Prediction")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 30)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type",
                                  ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        resting_bp = st.number_input("Resting BP (mm Hg)", 0, 300, 120)
        cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 600)
    with col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
        resting_ecg = st.selectbox("Resting ECG",
                                   ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        max_hr = st.number_input("Maximum Heart Rate", 60, 202, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0)
        st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
    st.markdown("</div>", unsafe_allow_html=True)

    # Encode
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    input_data = pd.DataFrame({
        'Age': [age], 'Sex': [sex], 'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp], 'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs], 'RestingECG': [resting_ecg],
        'MaxHR': [max_hr], 'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak], 'ST_Slope': [st_slope]
    })

    if st.button("🔮 Predict Now"):
        st.session_state["input_data"] = input_data
        st.session_state["show_result"] = True
        st.success("✅ Prediction Data Saved! Go to '📊 Prediction Result' tab to view results.")

# ================================================================
# TAB 2 — PREDICTION RESULT
# ================================================================
with tab2:
    st.header("📊 Prediction Result")
    if "show_result" not in st.session_state or not st.session_state["show_result"]:
        st.info("⚠️ Please go to the Predict tab and click 'Predict Now' first.")
    else:
        input_data = st.session_state["input_data"]
        modelnames = [
            'decision_tree_model.pkl', 'LogisticRegressionmodel.pkl',
            'random_forest_model.pkl', 'svm_model.pkl'
        ]
        algonames = ["Decision Tree", "Logistic Regression", "Random Forest", "SVM"]

        predictions = []
        for m in modelnames:
            try:
                model = pickle.load(open(m, 'rb'))
                predictions.append(model.predict(input_data))
            except Exception:
                st.warning(f"⚠️ Failed to load {m}")
                predictions.append([0])

        risk_score = np.mean([r[0] for r in predictions]) * 100

        st.subheader("🧪 Heart Disease Risk (%)")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Heart Disease Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red"},
                "steps": [
                    {"range": [0, 40], "color": "lightgreen"},
                    {"range": [40, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"}
                ]
            }
        ))
        st.plotly_chart(gauge, use_column_width=True)

        st.subheader("📋 Model-wise Prediction Results")
        for i, res in enumerate(predictions):
            if res[0] == 0:
                st.success(f"✅ {algonames[i]} → No Heart Disease")
            else:
                st.error(f"❌ {algonames[i]} → Heart Disease Detected")

# ================================================================
# TAB 3 — BULK PREDICT
# ================================================================
with tab3:
    st.header("📂 Bulk Prediction System")
    file = st.file_uploader("Upload CSV (11 Features or with HeartDisease column)", type=["csv"])

    if file:
        df = pd.read_csv(file)
        if "HeartDisease" in df.columns:
            df = df.drop("HeartDisease", axis=1)
        try:
            model = pickle.load(open('LogisticRegressionmodel.pkl', 'rb'))
            df["Prediction"] = model.predict(df)
            df["Prediction"] = df["Prediction"].map({0: "No Heart Disease", 1: "Heart Disease"})
            st.success("✅ Predictions Completed!")
            st.write(df)
            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download Results CSV", csv, "results.csv", "text/csv")
        except Exception:
            st.error("❌ Failed to load model for bulk prediction.")
    else:
        st.info("Please upload a CSV file with correct columns to predict in bulk.")

# ================================================================
# TAB 4 — MODEL INFO
# ================================================================
with tab4:
    st.header("📈 Model Accuracy Overview")
    acc = {
        "Decision Tree": 80.9,
        "Logistic Regression": 85.8,
        "Random Forest": 84.2,
        "SVM": 84.2
    }
    df = pd.DataFrame({"Model": list(acc.keys()), "Accuracy": list(acc.values())})
    fig = px.bar(df, x="Model", y="Accuracy", color="Accuracy", text="Accuracy")
    st.plotly_chart(fig)

# ================================================================
# TAB 5 — ECG using ONNX
# ================================================================
with tab5:
    st.header("🫀 ECG Image Diagnosis (Cloud-Compatible)")

    try:
        eff_session = ort.InferenceSession("efficientnet_ecg_model.onnx")
        hybrid_session = ort.InferenceSession("hybrid_ecg_model.onnx")
        st.success("✅ ECG ONNX models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load ONNX models: {e}")
        st.stop()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    uploaded_image = st.file_uploader("📤 Upload ECG Image", type=["jpg","png","jpeg"])
    class_labels = ["Normal", "Myocardial Infarction", "Abnormal Heartbeat", "History of MI"]
    medical_messages = {
        "Normal": "✅ Normal ECG — no abnormalities detected. Stay healthy!",
        "Myocardial Infarction": "⚠️ Myocardial Infarction detected! Immediate medical consultation required.",
        "Abnormal Heartbeat": "💓 Irregular heartbeat detected — consult a cardiologist.",
        "History of MI": "📋 Past history of heart problem — maintain medication and check-ups."
    }

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded ECG Image", use_column_width=True)
        img_tensor = transform(image).unsqueeze(0).numpy()
        eff_output = eff_session.run(None, {eff_session.get_inputs()[0].name: img_tensor})[0]
        hybrid_output = hybrid_session.run(None, {hybrid_session.get_inputs()[0].name: img_tensor})[0]
        eff_pred = np.argmax(eff_output, axis=1)[0]
        hybrid_pred = np.argmax(hybrid_output, axis=1)[0]

        st.subheader("🩺 Model Predictions")
        st.write("EfficientNet:", class_labels[eff_pred])
        st.write("Hybrid Model:", class_labels[hybrid_pred])

        if eff_pred == hybrid_pred:
            final_pred = class_labels[eff_pred]
            st.success(f"✅ Final Diagnosis: {final_pred}")
            st.markdown(medical_messages[final_pred])
        else:
            st.warning("⚠️ Models disagree — manual review advised")
            st.write(f"EfficientNet: {class_labels[eff_pred]}, Hybrid: {class_labels[hybrid_pred]}")
            for pred in [class_labels[eff_pred], class_labels[hybrid_pred]]:
                if pred != "Normal":
                    st.warning(medical_messages[pred])
