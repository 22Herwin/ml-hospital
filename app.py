from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os
import datetime
import time
import json
import requests
from typing import Any, Dict, List, Optional, Tuple

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Get DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Allow fallback when API key is missing; do not stop the app
USE_DEEPSEEK = bool(DEEPSEEK_API_KEY)
if not USE_DEEPSEEK:
    st.sidebar.warning("DEEPSEEK_API_KEY not found — using fallback clinical rules (AI unavailable)")
else:
    st.sidebar.success("DeepSeek API key loaded")

# Initialize session state
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = None
if 'admission_complete' not in st.session_state:
    st.session_state.admission_complete = False
if 'last_admission_id' not in st.session_state:
    st.session_state.last_admission_id = None
# track DeepSeek availability so we don't keep calling after a 402/error
if 'deepseek_unavailable' not in st.session_state:
    st.session_state['deepseek_unavailable'] = False

# Safe rerun helper
def safe_rerun():
    rerun = getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()
    else:
        st.session_state['_safe_rerun_toggle'] = not st.session_state.get('_safe_rerun_toggle', False)

st.set_page_config(page_title='AI Hospital Management System', layout='wide')

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
STOCK_CSV = os.path.join(DATA_DIR, 'medicine_stock.csv')
PATIENTS_LOG = os.path.join(DATA_DIR, 'admission_log.csv')
os.makedirs(DATA_DIR, exist_ok=True)

# Mock diagnosis mapping for fallback
DIAGNOSIS_MAPPING = {
    'D01': {'name': 'Pneumonia', 'medicines': ['Amoxicillin 500mg', 'Azithromycin 250mg']},
    'D02': {'name': 'Hypertension', 'medicines': ['Amlodipine 5mg', 'Lisinopril 10mg']},
    'D03': {'name': 'Diabetes', 'medicines': ['Metformin 500mg', 'Insulin Glargine']},
    'D04': {'name': 'Influenza', 'medicines': ['Oseltamivir 75mg', 'Paracetamol 500mg']},
    'D05': {'name': 'Gastroenteritis', 'medicines': ['Ondansetron 4mg', 'Loperamide 2mg']},
    'D06': {'name': 'Back Pain', 'medicines': ['Ibuprofen 400mg', 'Acetaminophen 650mg']}
}

# Stock management functions
@st.cache_resource
def load_stock():
    """Load medicine stock from CSV with fallback initialization"""
    try:
        if os.path.exists(STOCK_CSV):
            df = pd.read_csv(STOCK_CSV)
            df['stock'] = pd.to_numeric(df['stock'], errors='coerce').fillna(0).astype(int)
            return df
        else:
            # Create default stock if file doesn't exist
            meds = []
            for v in DIAGNOSIS_MAPPING.values():
                meds.extend(v.get('medicines', []))
            meds = sorted(list(set(meds)))
            df = pd.DataFrame({'medicine_name': meds, 'stock': [10]*len(meds)})
            df.to_csv(STOCK_CSV, index=False)
            return df
    except Exception as e:
        st.error(f"Error loading stock: {str(e)}")
        return pd.DataFrame(columns=['medicine_name', 'stock'])

def save_stock(df):
    """Save updated stock to CSV"""
    try:
        df.to_csv(STOCK_CSV, index=False)
        load_stock.clear()  # Clear cache
        return True
    except Exception as e:
        st.error(f"Error saving stock: {str(e)}")
        return False

# DeepSeek AI integration
def analyze_with_deepseek(features: dict) -> dict:
    """
    Analyze patient features using DeepSeek API
    Returns structured JSON response with diagnosis, recommendations, and rationale
    """
    # If API key not configured or previously marked unavailable, use fallback immediately
    if not USE_DEEPSEEK or st.session_state.get('deepseek_unavailable', False):
        return fallback_analysis(features)

    # Create clinical note from features
    clinical_note = f"""
    Patient Clinical Summary:
    Age: {features['age']} years, BMI: {features['bmi']}
    Vital Signs:
    - Blood Pressure: {features['blood_pressure_sys']}/{features['blood_pressure_dia']} mmHg
    - Heart Rate: {features['heart_rate']} bpm
    - Temperature: {features['temperature']} °C
    Symptoms:
    - Cough: {'Yes' if features['symptom_cough'] else 'No'}
    - Fever: {'Yes' if features['symptom_fever'] else 'No'}
    - Breathlessness: {'Yes' if features['symptom_breathless'] else 'No'}
    Comorbidities:
    - Diabetes: {'Yes' if features['comorbidity_diabetes'] else 'No'}
    - Hypertension: {'Yes' if features['comorbidity_hypertension'] else 'No'}
    Lab Results:
    - WBC Count: {features['lab_wbc']} x10^9/L
    - CRP Level: {features['lab_crp']} mg/L
    Clinical Severity Score: {features['severity_score']}
    """
    
    # Prepare AI prompt
    prompt = f"""
    You are an expert clinical decision support system. Analyze this patient case and provide a structured response in JSON format with these EXACT keys:
    - icd10_code: Primary ICD-10 diagnosis code (e.g., "J18.9")
    - diagnosis_name: Full diagnosis name matching ICD-10 code
    - inpatient: Boolean (true/false) for hospitalization recommendation
    - estimated_stay_days: Integer for estimated hospital days if admitted
    - ward_type: String describing ward type (e.g., "General", "ICU", "Cardiac")
    - recommended_medicines: List of 2-3 specific medication recommendations with dosages
    - rationale: Brief clinical justification for recommendations
    
    PATIENT DATA:
    {clinical_note}
    
    IMPORTANT: Return ONLY valid JSON with no additional text or explanation.
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    payload = {
        "model": "deepseek-r1",
        "messages": [
            {"role": "system", "content": "You are a clinical decision support AI."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    try:
        with st.spinner("DeepSeek AI analyzing patient case..."):
            response = requests.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

        # Handle insufficient balance explicitly: mark unavailable and fallback
        if response.status_code == 402:
            st.sidebar.error("DeepSeek API error: Insufficient balance. Switching to fallback rules.")
            st.session_state['deepseek_unavailable'] = True
            return fallback_analysis(features)

        if response.status_code != 200:
            st.error(f"DeepSeek API error ({response.status_code}): {response.text}")
            # Return a dict via the fallback rule-based analysis to satisfy the declared return type
            return fallback_analysis(features)

        # Extract AI response
        ai_response = response.json()['choices'][0]['message']['content']
        
        # Clean and parse JSON response
        json_start = ai_response.find('{')
        json_end = ai_response.rfind('}') + 1
        if json_start == -1 or json_end == -1:
            raise ValueError("Invalid JSON format in AI response")
        
        cleaned_json = ai_response[json_start:json_end]
        return json.loads(cleaned_json)
    
    except Exception as e:
        st.error(f"DeepSeek analysis failed: {str(e)}")
        # Fallback to rule-based system
        return fallback_analysis(features)

def fallback_analysis(features: dict) -> dict:
    """Fallback rule-based analysis when AI fails"""
    st.warning("Using fallback clinical rules (AI unavailable)")
    
    # Simple rule-based diagnosis
    if features['symptom_cough'] and features['symptom_fever'] and features['lab_wbc'] > 11:
        diagnosis = {'code': 'J18.9', 'name': 'Pneumonia', 'meds': ['Amoxicillin 500mg', 'Azithromycin 250mg']}
    elif features['comorbidity_hypertension'] and features['blood_pressure_sys'] > 160:
        diagnosis = {'code': 'I10', 'name': 'Hypertension', 'meds': ['Amlodipine 5mg', 'Lisinopril 10mg']}
    elif features['comorbidity_diabetes'] and features['lab_crp'] > 20:
        diagnosis = {'code': 'E11.9', 'name': 'Type 2 Diabetes', 'meds': ['Metformin 500mg', 'Insulin Glargine']}
    else:
        diagnosis = {'code': 'R50.9', 'name': 'Fever of unknown origin', 'meds': ['Paracetamol 500mg', 'Ibuprofen 400mg']}
    
    # Simple admission rules
    inpatient = features['severity_score'] >= 10
    stay_days = 5 if inpatient else 0
    ward = "ICU" if features['severity_score'] >= 15 else "General"
    
    return {
        "icd10_code": diagnosis['code'],
        "diagnosis_name": diagnosis['name'],
        "inpatient": inpatient,
        "estimated_stay_days": stay_days,
        "ward_type": ward,
        "recommended_medicines": diagnosis['meds'],
        "rationale": "Fallback rule-based analysis due to AI unavailability"
    }

# Calculate severity score (same logic as before)
def calc_severity_score(features: dict) -> int:
    score = 0
    # Vital signs
    if features['blood_pressure_sys'] > 180 or features['blood_pressure_sys'] < 90: score += 2
    if features['blood_pressure_dia'] > 120 or features['blood_pressure_dia'] < 60: score += 2
    if features['heart_rate'] > 120 or features['heart_rate'] < 50: score += 2
    if features['temperature'] > 39.0: score += 2
    
    # Symptoms
    if features['symptom_cough'] or features['symptom_fever'] or features['symptom_breathless']: score += 1
    if features['symptom_cough'] and features['symptom_fever']: score += 1
    if features['symptom_cough'] and features['symptom_breathless']: score += 1
    if features['symptom_fever'] and features['symptom_breathless']: score += 2
    
    # Lab values
    if features['lab_wbc'] > 15.0: score += 3
    elif features['lab_wbc'] > 11.0: score += 2
    if features['lab_crp'] > 50: score += 3
    elif features['lab_crp'] > 20: score += 2
    elif features['lab_crp'] > 10: score += 1
    
    # Comorbidities
    if features['comorbidity_diabetes']: score += 1
    if features['comorbidity_hypertension']: score += 1
    if features['age'] > 65: score += 1
    
    return score

# Sidebar configuration
st.sidebar.header('System Information')
# indicate DeepSeek availability if known
if st.session_state.get('deepseek_unavailable', False):
    st.sidebar.warning("DeepSeek unavailable (insufficient balance or blocked). Using fallback rules.")
else:
    st.sidebar.success(f"DeepSeek API Connected ({DEEPSEEK_BASE_URL})")
st.sidebar.info("""
**AI-Powered Hospital System**
- No ML models required
- Real-time clinical decision support
- Fallback rules when AI unavailable
""")

# Main content
st.title('AI Hospital Management System (DeepSeek R1)')
st.markdown("### Intelligent Patient Admission & Resource Management")

# Patient input form
with st.form('patient_form'):
    st.subheader("Patient Clinical Profile")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pid = st.text_input('Patient ID', value=f"P{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
        age = st.number_input('Age', min_value=0, max_value=120, value=45)
        sex = st.selectbox('Biological Sex', ['M', 'F', 'Other'])
        bmi = st.number_input('BMI', value=24.0, step=0.1, min_value=10.0, max_value=60.0)
    
    with col2:
        st.subheader("Vital Signs")
        bp_sys = st.number_input('Systolic BP (mmHg)', value=120, min_value=70, max_value=250)
        bp_dia = st.number_input('Diastolic BP (mmHg)', value=80, min_value=40, max_value=150)
        hr = st.number_input('Heart Rate (bpm)', value=78, min_value=30, max_value=200)
        temp = st.number_input('Temperature (°C)', value=36.7, step=0.1, min_value=30.0, max_value=42.0)
    
    with col3:
        st.subheader("Clinical Indicators")
        cough = st.checkbox('Cough')
        fever = st.checkbox('Fever (>38°C)')
        breathless = st.checkbox('Breathlessness')
        diabetes = st.checkbox('Diabetes')
        hypertension = st.checkbox('Hypertension')
        wbc = st.number_input('WBC Count (10^9/L)', value=7.0, step=0.1, min_value=0.0, max_value=50.0)
        crp = st.number_input('CRP Level (mg/L)', value=5.0, step=0.1, min_value=0.0, max_value=300.0)
    
    submitted = st.form_submit_button('Analyze with DeepSeek AI', type='primary')

if submitted:
    # Calculate severity score
    features = {
        'age': age, 'bmi': bmi,
        'blood_pressure_sys': bp_sys, 'blood_pressure_dia': bp_dia,
        'heart_rate': hr, 'temperature': temp,
        'symptom_cough': int(cough), 'symptom_fever': int(fever), 'symptom_breathless': int(breathless),
        'comorbidity_diabetes': int(diabetes), 'comorbidity_hypertension': int(hypertension),
        'lab_wbc': wbc, 'lab_crp': crp
    }
    severity_score = calc_severity_score(features)
    features['severity_score'] = severity_score
    
    # Store patient data
    st.session_state.current_patient = {
        'pid': pid,
        'features': features,
        'severity_score': severity_score,
        'timestamp': datetime.datetime.now()
    }
    
    # Analyze with DeepSeek AI (or fallback if previously flagged)
    if st.session_state.get('deepseek_unavailable', False):
        ai_result = fallback_analysis(features)
    else:
        ai_result = analyze_with_deepseek(features)
    
    if ai_result:
        st.session_state.current_patient['ai_result'] = ai_result
        st.session_state.current_patient['diagnosis_code'] = ai_result.get('icd10_code', 'Unknown')
        st.session_state.current_patient['diagnosis_name'] = ai_result.get('diagnosis_name', 'Unknown')
    
    st.session_state.admission_complete = False
    safe_rerun()

# Display analysis results
if st.session_state.current_patient and not st.session_state.admission_complete:
    patient = st.session_state.current_patient
    features = patient['features']
    ai_result = patient.get('ai_result', {})
    
    st.subheader("DeepSeek AI Clinical Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Patient Features Analyzed:**")
        st.json(features)
    
    with col2:
        st.metric("Clinical Severity Score", patient['severity_score'], 
                 delta="Critical" if patient['severity_score'] >= 15 else 
                       "High Risk" if patient['severity_score'] >= 10 else 
                       "Moderate Risk" if patient['severity_score'] >= 5 else "Low Risk",
                 delta_color="inverse")
        
        # Severity interpretation
        score = patient['severity_score']
        if score >= 15:
            st.error("Critical Condition - Requires immediate intervention")
        elif score >= 10:
            st.warning("High Risk - Close monitoring needed")
        elif score >= 5:
            st.info("Moderate Risk - Regular monitoring recommended")
        else:
            st.success("Low Risk - Outpatient management appropriate")
    
    # AI Results Display
    st.markdown("---")
    st.subheader("AI Clinical Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if ai_result:
            st.success(f"**Diagnosis:** {ai_result.get('diagnosis_name', 'Unknown')} ({ai_result.get('icd10_code', 'Unknown')})")
            st.info(f"**Hospitalization Recommended:** {'Yes' if ai_result.get('inpatient') else 'No'}")
            if ai_result.get('inpatient'):
                st.warning(f"**Recommended Ward:** {ai_result.get('ward_type', 'General')}")
                st.info(f"**Estimated Stay:** {ai_result.get('estimated_stay_days', 3)} days")
        else:
            st.error("AI analysis failed - using fallback rules")
    
    with col2:
        st.subheader("Recommended Medications")
        if ai_result and ai_result.get('recommended_medicines'):
            for med in ai_result['recommended_medicines']:
                st.write(f"- {med}")
        else:
            st.write("No recommendations available")
    
    # Rationale section
    with st.expander("Clinical Rationale (AI Explanation)"):
        if ai_result:
            st.write(ai_result.get('rationale', 'No rationale provided'))
        else:
            st.write("Fallback rule-based analysis used")
    
    # Admission workflow
    if ai_result and ai_result.get('inpatient'):
        st.markdown("---")
        st.subheader("Admission Workflow")
        
        stock_df = load_stock()
        recommended_meds = ai_result.get('recommended_medicines', [])
        
        # Filter available medicines
        available_meds = stock_df[
            stock_df['medicine_name'].apply(lambda x: any(med in x for med in recommended_meds)) & 
            (stock_df['stock'] > 0)
        ]
        
        if available_meds.empty:
            st.warning("No recommended medicines in stock. Please replenish inventory.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                selected_med = st.selectbox(
                    'Select Medication to Assign', 
                    available_meds['medicine_name'].tolist(),
                    help="Choose from available in-stock medications"
                )
            
            with col2:
                current_stock = available_meds[available_meds['medicine_name'] == selected_med]['stock'].values[0]
                qty = st.number_input(
                    'Quantity to Assign', 
                    min_value=1, 
                    max_value=int(current_stock), 
                    value=1,
                    help=f"Available stock: {current_stock}"
                )
            
            if st.button('CONFIRM ADMISSION', type='primary', use_container_width=True):
                # Create admission record
                admission_data = {
                    'patient_id': patient['pid'],
                    'admit_time': patient['timestamp'].isoformat(),
                    'ward_type': ai_result.get('ward_type', 'General'),
                    'estimated_days': ai_result.get('estimated_stay_days', 3),
                    'med_used': selected_med,
                    'qty': int(qty),
                    'diagnosis_code': ai_result.get('icd10_code', 'Unknown'),
                    'diagnosis_name': ai_result.get('diagnosis_name', 'Unknown'),
                    'severity_score': patient['severity_score']
                }
                
                # Save admission to log
                os.makedirs(DATA_DIR, exist_ok=True)
                admission_df = pd.DataFrame([admission_data])
                if os.path.exists(PATIENTS_LOG):
                    admission_df.to_csv(PATIENTS_LOG, mode='a', header=False, index=False)
                else:
                    admission_df.to_csv(PATIENTS_LOG, index=False)
                
                # Update stock
                stock_df.loc[stock_df['medicine_name'] == selected_med, 'stock'] -= qty
                save_stock(stock_df)
                
                st.success(f"Patient {patient['pid']} admitted to {admission_data['ward_type']} ward")
                st.session_state.last_admission_id = patient['pid']
                st.session_state.admission_complete = True
                safe_rerun()

# Inventory management section
st.markdown("---")
st.subheader("Medicine Inventory Management")

stock_df = load_stock()
st.dataframe(stock_df)

col1, col2 = st.columns(2)

with col1:
    if st.button('Refresh Inventory'):
        load_stock.clear()
        safe_rerun()

with col2:
    if st.button('Replenish Low Stock (+10 units)'):
        low_stock = stock_df[stock_df['stock'] < 5]
        if not low_stock.empty:
            for idx in low_stock.index:
                # Ensure the stored value is numeric before adding
                current_val = pd.to_numeric(stock_df.at[idx, 'stock'], errors='coerce')
                if pd.isna(current_val):
                    current_val = 0
                stock_df.at[idx, 'stock'] = int(current_val) + 10
            save_stock(stock_df)
            st.success(f"Replenished {len(low_stock)} low-stock items")
        else:
            st.info("All items have sufficient stock")

# Admission history
st.markdown("---")
st.subheader("Recent Admissions")

if os.path.exists(PATIENTS_LOG):
    try:
        log_df = pd.read_csv(PATIENTS_LOG)
        st.dataframe(log_df.tail(5))
    except Exception as e:
        st.error(f"Error loading admission log: {str(e)}")
else:
    st.info("No admission history available yet")

st.markdown("---")
st.caption("AI Hospital Management System • Powered by DeepSeek R1 • © 2025")