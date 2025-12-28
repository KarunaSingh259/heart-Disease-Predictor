import plotly.express as px
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predict",
    "📊 Prediction Result",
    "📂 Bulk Predict",
    "📄 Model Info",
    "🫀 ECG Image Test"
])

# ==============================
# TAB 1 – INPUT
# ==============================
with tab1:
    st.header("🔍 Enter Patient Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        rbp = st.number_input("Resting BP", 0, 300, 120)
        chol = st.number_input("Cholesterol", 0, 600)
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar", ["<=120", ">120"])
        ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LVH"])
        maxhr = st.number_input("Max Heart Rate", 60, 220, 150)
        angina = st.selectbox("Exercise Angina", ["Yes", "No"])
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
        slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    # Convert categorical to numeric
    input_df = pd.DataFrame({
        "Age": [age],
        "Sex": [0 if sex == "Male" else 1],
        "ChestPainType": [["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp)],
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
        st.session_state["data"] = input_df
        st.session_state["show"] = True
        st.success("Data saved. Go to Prediction Result tab.")

# ==============================
# TAB 2 – PREDICTION RESULT (DASHBOARD STYLE)
# ==============================
with tab2:
    st.header("📊 Prediction Result")

    if "show" not in st.session_state:
        st.info("Please predict first from Tab 1.")
        st.stop()

    input_data = st.session_state["data"]

    # Collect predictions from all models
    preds = []
    for name, model in ml_models.items():
        pred = model.predict(input_data)[0]
        preds.append(pred)

    if not preds:
        st.error("No predictions available.")
        st.stop()

    # Calculate average risk percentage
    risk = np.mean(preds) * 100

    # ==============================
    # COLORFUL GAUGE
    # ==============================
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk,
        title={"text": "Heart Disease Risk (%)", "font": {"size": 24, "color": "#1f2c56"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkred"},
            "steps": [
                {"range": [0, 25], "color": "lightgreen"},
                {"range": [25, 50], "color": "yellow"},
                {"range": [50, 75], "color": "orange"},
                {"range": [75, 100], "color": "red"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": risk
            }
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

    # ==============================
    # MODEL-WISE COLORFUL BOXES
    # ==============================
    st.subheader("📋 Model-wise Prediction Results")

    for name, pred in zip(ml_models.keys(), preds):
        if pred == 1:
            st.markdown(
                f"""
                <div style='background-color:#8B0000; color:white; padding:12px; border-radius:8px; margin-bottom:6px; font-size:16px;'>
                    ❌ {name} → Heart Disease Detected
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='background-color:#228B22; color:white; padding:12px; border-radius:8px; margin-bottom:6px; font-size:16px;'>
                    ✅ {name} → No Heart Disease
                </div>
                """,
                unsafe_allow_html=True
            )

# ==============================
# TAB 3 – BULK PREDICTION
# ==============================
with tab3:
    st.header("📂 Bulk Prediction")
    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df = pd.read_csv(file)

        # Drop target column if it exists
        if "HeartDisease" in df.columns:
            df = df.drop("HeartDisease", axis=1)

        # Use cached Logistic Regression model
        model = ml_models.get("Logistic Regression")  # ✅ No need for load_pickle_model
        if model:
            df["Prediction"] = model.predict(df)
            df["Prediction"] = df["Prediction"].map({0: "No Disease", 1: "Disease"})
            st.write(df)
        else:
            st.error("Logistic Regression model not loaded!")
# ==============================
# TAB 4 – MODEL INFO
# ==============================
with tab4:
    st.header("📈 Model Accuracy")
    acc = {
        "Decision Tree": 80.9,
        "Logistic Regression": 85.8,
        "Random Forest": 84.2,
        "SVM": 84.2,
        "Gridrf": 89.7
    }
    df = pd.DataFrame(acc.items(), columns=["Model", "Accuracy"])
    st.plotly_chart(px.bar(df, x="Model", y="Accuracy"))


# ==============================
# TAB 5 – ECG IMAGE TEST DASHBOARD
# ==============================
with tab5:
    st.header("🫀 ECG Image Diagnosis ")

    uploaded = st.file_uploader("Upload ECG Image", ["jpg", "png"])
    
    labels = ["Normal", "Myocardial Infarction", "Abnormal Heartbeat", "History of MI"]

    # Health suggestions for each class
    suggestions = {
        "Normal": "✅ ECG is normal. Maintain a healthy lifestyle and regular check-ups.",
        "Myocardial Infarction": "❌ Signs of a heart attack. Consult a cardiologist immediately!",
        "Abnormal Heartbeat": "⚠️ Irregular heartbeat detected. Monitor and consult a cardiologist.",
        "History of MI": "⚠️ Previous heart attack detected. Follow up with your doctor regularly."
    }

    if uploaded:
        # Display the original high-quality image
        img_original = Image.open(uploaded).convert("RGB")
        st.image(img_original, caption="Uploaded ECG", use_column_width=True)

        # Resize image for ONNX model
        img_resized = img_original.resize((224, 224))
        img_arr = np.array(img_resized).astype(np.float32) / 255.0
        img_arr = np.transpose(img_arr, (2, 0, 1))
        img_arr = np.expand_dims(img_arr, axis=0)

        # Run ONNX models
        eff_pred = eff_sess.run(None, {"input": img_arr})[0]
        hyb_pred = hyb_sess.run(None, {"input": img_arr})[0]

        eff_class = labels[np.argmax(eff_pred)]
        hyb_class = labels[np.argmax(hyb_pred)]

        # ==============================
        # Colorful prediction boxes
        # ==============================
        st.subheader("📋 Model-wise Prediction Results")

        for model_name, pred_class in zip(["EfficientNet", "Hybrid Model"], [eff_class, hyb_class]):
            if pred_class == "Normal":
                st.markdown(
                    f"""
                    <div style='background-color:#228B22; color:white; padding:12px; border-radius:8px; margin-bottom:6px; font-size:16px;'>
                        ✅ {model_name} → {pred_class}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style='background-color:#8B0000; color:white; padding:12px; border-radius:8px; margin-bottom:6px; font-size:16px;'>
                        ❌ {model_name} → {pred_class}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ==============================
        # Health suggestions
        # ==============================
        st.subheader("📝 Health Suggestions")
        st.info(f"EfficientNet: {suggestions[eff_class]}")
        st.info(f"Hybrid Model: {suggestions[hyb_class]}")








