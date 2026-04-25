from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, session
import os
import re
import random
import pickle
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import PyPDF2
import cv2
import tempfile

# Load ML Model
try:
    with open('models/ml/LogisticRegressionmodel.pkl', 'rb') as f:
        ml_model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    ml_model = None

# ----------------- DEEP LEARNING MODEL SETUP -----------------
import torch
import torch.nn as nn
from torchvision import models as tv_models
from torchvision.models.video import r2plus1d_18
from torchvision import transforms
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

dl_model = None
try:
    print("Loading Simple CNN ECG Model...")
    dl_model = SimpleCNN(num_classes=4)
    state_dict = torch.load('models/dl/ecg_model.pth', map_location='cpu')
    dl_model.load_state_dict(state_dict)
    dl_model.eval()
    print("Simple CNN ECG Model loaded successfully.")
except Exception as e:
    print(f"Error loading DL model: {e}")
    dl_model = None

# Transforms for ECG (Updated for SimpleCNN 128x128)
ecg_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

ecg_classes = [
    'Myocardial Infarction',
    'History of MI',
    'Abnormal Heartbeat',
    'Normal Person'
]
# -------------------------------------------------------------

# ----------------- ECHO MODEL SETUP -----------------
echo_model = None
try:
    print("Loading Echo Video Model (R2Plus1D-18)... this may take a moment.")
    # Initialize R2Plus1D-18 architecture
    echo_model = r2plus1d_18(weights=None)
    # Modify the fc layer to match the saved checkpoint (Sequential with Dropout and Linear)
    echo_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    
    # Load the checkpoint
    checkpoint = torch.load('models/dl/echo_classifier.pth', map_location='cpu')
    if 'model_state_dict' in checkpoint:
        echo_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        echo_model.load_state_dict(checkpoint)
        
    echo_model.eval()
    print("Echo Model loaded successfully.")
except Exception as e:
    print(f"Error loading Echo model: {e}")
    echo_model = None

# Transforms for Echo (Updated for R2Plus1D 112x112)
echo_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989])
])

def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        return None
    
    # Extract evenly spaced frames
    step = max(1, total_frames // num_frames)
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i * step, total_frames - 1))
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = echo_transform(frame)
        frames.append(tensor_frame)
        
    cap.release()
    
    # Pad if we didn't get enough frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else torch.zeros((3, 224, 224)))
        
    return torch.stack(frames)
# -------------------------------------------------------------

# Encoders
sex_map = {'M': 1, 'F': 0}
chest_map = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}
ecg_map = {'LVH': 0, 'Normal': 1, 'ST': 2}
angina_map = {'N': 0, 'Y': 1}
slope_map = {'Down': 0, 'Flat': 1, 'Up': 2}

app = Flask(__name__)
app.secret_key = 'cardio_ai_secret_key'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/single_prediction', methods=['GET', 'POST'])
def single_prediction():
    if request.method == 'POST':
        if not ml_model:
            return jsonify({'error': 'Model not loaded'}), 500
            
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            age = float(data.get('age', 0))
            sex = sex_map.get(data.get('sex', 'M'), 1)
            chest = chest_map.get(data.get('chest_pain', 'ASY'), 0)
            bp = float(data.get('resting_bp', 0))
            chol = float(data.get('cholesterol', 0))
            fasting = float(data.get('fasting_bs', 0))
            ecg = ecg_map.get(data.get('resting_ecg', 'Normal'), 1)
            hr = float(data.get('max_hr', 0))
            angina = angina_map.get(data.get('exercise_angina', 'N'), 0)
            oldpeak = float(data.get('oldpeak', 0.0))
            slope = slope_map.get(data.get('st_slope', 'Flat'), 1)
            
            features = np.array([[age, sex, chest, bp, chol, fasting, ecg, hr, angina, oldpeak, slope]])
            
            prediction = ml_model.predict(features)[0]
            probability = ml_model.predict_proba(features)[0][1] * 100
            
            risk_level = "High Risk" if prediction == 1 else "Low Risk"
            
            session['clinical_pred'] = {
                'risk_level': risk_level,
                'confidence': round(probability, 1)
            }
            
            return jsonify({
                'success': True,
                'risk_level': risk_level,
                'confidence': round(probability, 1)
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
            
    return render_template('single_prediction.html')

@app.route('/bulk_prediction', methods=['GET', 'POST'])
def bulk_prediction():
    if request.method == 'POST':
        if not ml_model:
            return jsonify({'error': 'Model not loaded'}), 500
            
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        try:
            df = pd.read_csv(file)
            display_df = df.copy()
            
            # Map categorical columns safely
            df['Sex'] = df['Sex'].map(sex_map).fillna(1)
            df['ChestPainType'] = df['ChestPainType'].map(chest_map).fillna(0)
            df['RestingECG'] = df['RestingECG'].map(ecg_map).fillna(1)
            df['ExerciseAngina'] = df['ExerciseAngina'].map(angina_map).fillna(0)
            df['ST_Slope'] = df['ST_Slope'].map(slope_map).fillna(1)
            
            # Make sure we only use the 11 required columns
            expected_cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
            features = df[expected_cols].values
            
            predictions = ml_model.predict(features)
            probabilities = ml_model.predict_proba(features)[:, 1] * 100
            
            results = []
            high_risk_count = 0
            moderate_risk_count = 0
            low_risk_count = 0
            
            for i in range(len(df)):
                ml_is_high_risk = int(predictions[i]) == 1
                
                bp_val = int(display_df['RestingBP'].iloc[i])
                chol_val = int(display_df['Cholesterol'].iloc[i])
                
                if bp_val > 140 or chol_val > 240:
                    final_risk = "High Risk"
                elif bp_val > 130 or chol_val > 200:
                    final_risk = "Moderate Risk"
                else:
                    final_risk = "High Risk" if ml_is_high_risk else "Low Risk"
                    
                if final_risk == "High Risk":
                    high_risk_count += 1
                elif final_risk == "Moderate Risk":
                    moderate_risk_count += 1
                else:
                    low_risk_count += 1
                    
                results.append({
                    'id': f"#{i+1:03d}",
                    'age': int(display_df['Age'].iloc[i]),
                    'sex': str(display_df['Sex'].iloc[i]),
                    'chest': str(display_df['ChestPainType'].iloc[i]),
                    'bp': bp_val,
                    'chol': chol_val,
                    'confidence': round(float(probabilities[i]), 1),
                    'risk': final_risk
                })
                
            return jsonify({
                'success': True,
                'total': len(df),
                'high_risk': high_risk_count,
                'moderate_risk': moderate_risk_count,
                'low_risk': low_risk_count,
                'results': results[:100] # Return top 100 for display max
            })
            
        except Exception as e:
            return jsonify({'error': f'Invalid CSV format or missing columns: {str(e)}'}), 400
            
    return render_template('bulk_prediction.html')

@app.route('/ecg_analysis', methods=['GET', 'POST'])
def ecg_analysis():
    if request.method == 'POST':
        if not dl_model:
            return jsonify({'error': 'DL Model not loaded'}), 500
            
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        try:
            image = Image.open(file).convert('RGB')
            tensor = ecg_transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = dl_model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            class_idx = predicted.item()
            conf_val = round(confidence.item() * 100, 1)
            predicted_class = ecg_classes[class_idx]
            
            risk_level = "Low Risk" if class_idx == 3 else "High Risk"
            
            # Clinical notes mapping (Aligned with new class order)
            notes_map = {
                0: "Possible myocardial infarction detected. The ST-segment elevation suggests acute injury to the myocardium.",
                1: "Signs of previous myocardial infarction present, such as pathological Q waves.",
                2: "Arrhythmia or abnormal heartbeat detected in the rhythm strip.",
                3: "Normal sinus rhythm. No significant ST-T wave abnormalities detected."
            }
            
            session['ecg_pred'] = {
                'predicted_class': predicted_class,
                'confidence': conf_val,
                'risk_level': risk_level,
                'clinical_note': notes_map[class_idx]
            }
            
            return jsonify({
                'success': True,
                'predicted_class': predicted_class,
                'confidence': conf_val,
                'risk_level': risk_level,
                'clinical_note': notes_map[class_idx]
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
            
    return render_template('ecg_analysis.html')

@app.route('/echo_analysis', methods=['GET', 'POST'])
def echo_analysis():
    if request.method == 'POST':
        if not echo_model:
            return jsonify({'error': 'Echo Model not loaded'}), 500
            
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400
            
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        try:
            # Save video temporarily for OpenCV
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                file.save(temp_video.name)
                temp_video_path = temp_video.name
                
            tensor_frames = extract_frames(temp_video_path, num_frames=16)
            os.remove(temp_video_path)
            
            if tensor_frames is None:
                return jsonify({'error': 'Could not extract frames from video'}), 400
                
            # Add batch dimension and permute to (Batch, Channels, Time, Height, Width)
            # Current shape from extract_frames: (16, 3, 112, 112)
            # Unsqueeze(0) -> (1, 16, 3, 112, 112)
            # Permute(0, 2, 1, 3, 4) -> (1, 3, 16, 112, 112)
            tensor_frames = tensor_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
            
            with torch.no_grad():
                # R2Plus1D model returns only classification logits
                outputs = echo_model(tensor_frames)
                
                # Process Classification
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                class_idx = predicted.item()
                conf_val = round(confidence.item() * 100, 1)
                
            # Mapping based on checkpoint: ['Normal', 'Abnormal']
            status = "Abnormal Findings" if class_idx == 1 else "Normal Echo"
            
            if class_idx == 1:
                interpretation = "The model detected abnormal wall motion or structural patterns consistent with compromised cardiac function. Further clinical evaluation is recommended."
            else:
                interpretation = "The echocardiogram shows normal cardiac structure and wall motion. No significant abnormalities were detected."
                
            session['echo_pred'] = {
                'status': status,
                'confidence': conf_val,
                'ejection_fraction': "N/A", # No longer provided by this model
                'interpretation': interpretation,
                'class_idx': class_idx
            }
                
            return jsonify({
                'success': True,
                'status': status,
                'confidence': conf_val,
                'ejection_fraction': "N/A",
                'interpretation': interpretation,
                'class_idx': class_idx
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
            
    return render_template('echo_analysis.html')

@app.route('/report_summary', methods=['GET', 'POST'])
def report_summary():
    if request.method == 'POST':
        if 'report' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['report']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if file:
            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            text = ""
            
            try:
                if ext == 'pdf':
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                elif ext in ['txt', 'csv']:
                    text = file.read().decode('utf-8')
                else:
                    return jsonify({'error': 'Unsupported file format'}), 400
            except Exception as e:
                return jsonify({'error': f'Error reading file: {str(e)}'}), 500
                
            # Basic NLP/Regex Extraction
            findings = []
            risk_score = 0
            
            text_lower = text.lower()
            
            # Cholesterol check
            chol_match = re.search(r'(?:cholesterol|ldl|hdl).*?(\d+)', text_lower)
            if chol_match:
                val = int(chol_match.group(1))
                if val > 200:
                    findings.append(f"Elevated Cholesterol ({val} mg/dL)")
                    risk_score += 2
                else:
                    findings.append(f"Normal Cholesterol ({val} mg/dL)")
            elif 'hyperlipidemia' in text_lower or 'high cholesterol' in text_lower:
                findings.append("History of Hyperlipidemia")
                risk_score += 2
                
            # BP check
            bp_match = re.search(r'(?:blood pressure|bp).*?(\d{2,3})\s*/\s*(\d{2,3})', text_lower)
            if bp_match:
                sys, dia = int(bp_match.group(1)), int(bp_match.group(2))
                if sys > 130 or dia > 80:
                    findings.append(f"Hypertension documented ({sys}/{dia} mmHg)")
                    risk_score += 2
                else:
                    findings.append(f"Normal Blood Pressure ({sys}/{dia} mmHg)")
            elif 'hypertension' in text_lower:
                findings.append("History of Hypertension")
                risk_score += 2
                
            # ECG/Echo keywords
            if 'hypertrophy' in text_lower:
                findings.append("Left ventricular hypertrophy")
                risk_score += 1
            if 'ischemia' in text_lower or 'infarction' in text_lower or 'st elevation' in text_lower:
                findings.append("Ischemia/Infarction patterns detected")
                risk_score += 3
            if 'angina' in text_lower:
                findings.append("Reports of Angina")
                risk_score += 2
            if 'family history' in text_lower and ('coronary' in text_lower or 'heart' in text_lower):
                findings.append("Family history of coronary artery disease")
                risk_score += 1
                
            if not findings:
                findings.append("No critical cardiac risk factors extracted from report.")
                
            # Determine Risk Level
            if risk_score >= 4:
                risk_level = "High Risk"
                risk_color = "#c53030" # red
                message = "High cardiac risk detected based on NLP extracted findings. Immediate medical intervention is highly advised."
            elif risk_score >= 2:
                risk_level = "Moderate Risk"
                risk_color = "#d69e2e" # yellow
                message = "Moderate cardiac risk detected. Please consult a cardiologist for further evaluation."
            else:
                risk_level = "Low Risk"
                risk_color = "#276749" # green
                message = "Low cardiac risk detected from the report. Maintain a healthy lifestyle."
                
            return jsonify({
                'success': True,
                'findings': findings,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'message': message,
                'clinical_pred': session.get('clinical_pred'),
                'ecg_pred': session.get('ecg_pred'),
                'echo_pred': session.get('echo_pred')
            })
            
    return render_template('report_summary.html', 
                           clinical_pred=session.get('clinical_pred'),
                           ecg_pred=session.get('ecg_pred'),
                           echo_pred=session.get('echo_pred'))

@app.route('/hospitals', methods=['GET', 'POST'])
def hospitals():
    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        state = data.get('state', '')
        pincode = data.get('pincode', 'N/A')
        if not pincode:
            pincode = 'N/A'
        
        if not state:
            return jsonify({'error': 'State is required'}), 400
            
        # Specific hospitals for each state
        state_hospitals = {
            "Delhi": [
                "AIIMS Cardiology Department",
                "Fortis Escorts Heart Institute",
                "Max Super Speciality Hospital, Saket",
                "Apollo Heart Institutes",
                "GB Pant Hospital"
            ],
            "Maharashtra": [
                "Asian Heart Institute, Mumbai",
                "Sir H. N. Reliance Foundation Hospital",
                "Kokilaben Dhirubhai Ambani Hospital",
                "Lilavati Hospital and Research Centre",
                "Fortis Hospital, Mulund"
            ],
            "Karnataka": [
                "Narayana Institute of Cardiac Sciences",
                "Manipal Hospital, HAL Airport Road",
                "Fortis Hospital, Bannerghatta",
                "Sri Jayadeva Institute of Cardiovascular Sciences",
                "Apollo Hospitals, Bengaluru"
            ],
            "Tamil Nadu": [
                "Apollo Main Hospital, Greams Road",
                "Madras Medical Mission",
                "Fortis Malar Hospital",
                "MIOT International",
                "Chettinad Health City"
            ],
            "Uttar Pradesh": [
                "Sanjay Gandhi Postgraduate Institute (SGPGIMS)",
                "Medanta Hospital, Lucknow",
                "Apollo Medics Super Speciality Hospital",
                "Sahara Hospital",
                "Jaypee Hospital, Noida"
            ],
            "Gujarat": [
                "U. N. Mehta Institute of Cardiology",
                "Zydus Hospitals, Ahmedabad",
                "Apollo Hospitals, Gandhinagar",
                "CIMS Hospital",
                "Sterling Hospitals"
            ],
            "West Bengal": [
                "RTIICS (Narayana Superspeciality)",
                "Apollo Multispeciality Hospitals",
                "Fortis Hospital, Anandapur",
                "BM Birla Heart Research Centre",
                "Medica Superspecialty Hospital"
            ]
        }
        
        localities = [
            "Civil Lines", "MG Road", "Main Highway", "City Center",
            "Sector 14", "Health Avenue", "Medical Enclave", "Jubilee Hills",
            "Park Street", "Gandhi Nagar"
        ]
        
        # Get exactly 5 hospitals for the selected state, or fallback if state not found
        selected_names = state_hospitals.get(state, [
            "City Heart Care", "Metro Cardiology Center", "National Heart Institute", 
            "Apex Super Speciality", "Premier Cardiac Hospital"
        ])
        
        num_results = 5
        results = []
        
        for i in range(num_results):
            name = selected_names[i]
            locality = random.choice(localities)
            dist = round(random.uniform(1.0, 15.0), 1)
            rating = round(random.uniform(4.0, 4.9), 1)
            
            # Format phone number for India
            phone = f"+91 {random.randint(70000, 99999)} {random.randint(10000, 99999)}"
            
            # Address format
            address = f"{random.randint(1, 100)}, {locality}, {state}"
            if pincode and pincode != 'N/A':
                address += f" {pincode}"
                
            results.append({
                'name': name,
                'address': address,
                'state': state,
                'pincode': pincode,
                'rating': rating,
                'distance': dist,
                'phone': phone
            })
            
        # Sort by distance
        results.sort(key=lambda x: x['distance'])
        
        return jsonify({
            'success': True,
            'hospitals': results,
            'count': len(results)
        })
        
    return render_template('hospitals.html')

@app.route('/model_info')
def model_info():
    return render_template('model_info.html')

if __name__ == '__main__':
    app.run(debug=True)