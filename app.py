import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
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
# SESSION STATE INIT
# ==============================
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "🔮 Predict"

# ==============================
# NAVIGATION
# ==============================
tab = st.radio(
    "Navigation",
    ["🔮 Predict", "📊 Prediction Result", "📂 Bulk Predict", "📄 Model Info", "🫀 ECG Image Test"],
    horizontal=True,
    index=[
        "🔮 Predict",
        "📊 Prediction Result",
        "📂 Bulk Predict",
        "📄 Model Info",
        "🫀 ECG Image Test"
    ].index(st.session_state.active_tab)
)

# ==============================
# LOAD ML MODELS
# ==============================
@st.cache_resource
def load_ml_models():
    return {
        "Decision Tree": pickle.load(open("decision_tree_model.pkl", "rb")),
        "Logistic Regression": pickle.load(open("LogisticRegressionmodel.pkl", "rb")),
        "Random Forest": pickle.load(open("random_forest_model.pkl", "rb")),
        "SVM": pickle.load(open("svm_model.pkl", "rb"))
    }

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
# TAB 1 – PREDICT
# ==============================
if tab == "🔮 Predict":
    st.header("🔍 Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
        )
        rbp = st.number_input("Resting BP", 0, 300, 120)
        chol = st.number_input("Cholesterol", 0, 600)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar", ["<=120", ">120"])
        ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LVH"])
        maxhr = st.number_input("Max Heart Rate", 60, 220, 150)
        angina = st.selectbox("Exercise Angina", ["Yes", "No"])
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
        slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    input_df = pd.DataFrame({
        "Age": [age],
        "Sex": [0 if sex == "Male" else 1],
        "ChestPainType": [
            ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp)
        ],
        "RestingBP": [rbp],
        "Cholesterol": [chol],
        "FastingBS": [1 if fbs == ">120" else 0],
        "RestingECG": [["Normal", "ST-T Abnormality", "LVH"].index(ecg)],
        "MaxHR": [maxhr],
        "ExerciseAngina": [1 if angina == "Yes" else 0],
        "Oldpeak": [oldpeak],
        "ST_Slope": [["Up", "Flat", "Down"].index(slope)]
    })

    if st.button("🔮 Predict Now"):
        st.session_state.data = input_df
        st.session_state.active_tab = "📊 Prediction Result"
        st.rerun()


# ==============================
# TAB 2 – PREDICTION RESULT
# ==============================
elif tab == "📊 Prediction Result":
    st.header("📊 Prediction Result")

    if "data" not in st.session_state:
        st.warning("Please enter patient details first.")
        st.stop()

    input_data = st.session_state.data
    preds = [model.predict(input_data)[0] for model in ml_models.values()]
    risk = np.mean(preds) * 100

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={"text": "Heart Disease Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkred"},
            "steps": [
                {"range": [0, 25], "color": "lightgreen"},
                {"range": [25, 50], "color": "yellow"},
                {"range": [50, 75], "color": "orange"},
                {"range": [75, 100], "color": "red"}
            ]
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

    st.subheader("📋 Model-wise Results")
    for name, pred in zip(ml_models.keys(), preds):
        if pred == 1:
            st.error(f"{name}: Heart Disease Detected")
        else:
            st.success(f"{name}: No Heart Disease")

# ==============================
# TAB 3 – BULK PREDICT
# ==============================
elif tab == "📂 Bulk Predict":
    st.header("📂 Bulk Prediction")

    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        if "HeartDisease" in df.columns:
            df.drop("HeartDisease", axis=1, inplace=True)

        model = ml_models["Logistic Regression"]
        df["Prediction"] = model.predict(df)
        df["Prediction"] = df["Prediction"].map({0: "No Disease", 1: "Disease"})
        st.dataframe(df)

# ==============================
# TAB 4 – MODEL INFO
# ==============================
elif tab == "📄 Model Info":
    st.header("📈 Model Accuracy")

    acc = {
        "Decision Tree": 80.9,
        "Logistic Regression": 85.8,
        "Random Forest": 84.2,
        "SVM": 84.2,
        "Gridrf":89.5
    }

    df_acc = pd.DataFrame(acc.items(), columns=["Model", "Accuracy"])
    st.plotly_chart(
        px.bar(df_acc, x="Model", y="Accuracy", color="Accuracy",
               color_continuous_scale="Viridis"),
        use_container_width=True
    )

# ==============================
# TAB 5 – ECG IMAGE TEST
# ==============================
elif tab == "🫀 ECG Image Test":
    st.header("🫀 ECG Image Diagnosis")

    uploaded = st.file_uploader("Upload ECG Image", ["jpg", "png"])
    labels = ["Normal", "Myocardial Infarction", "Abnormal Heartbeat", "History of MI"]

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True)

        img = img.resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)

        eff_pred = eff_sess.run(None, {"input": arr})[0]
        hyb_pred = hyb_sess.run(None, {"input": arr})[0]

        st.success(f"EfficientNet: {labels[np.argmax(eff_pred)]}")
        st.success(f"Hybrid Model: {labels[np.argmax(hyb_pred)]}")


