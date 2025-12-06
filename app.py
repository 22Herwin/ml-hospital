"""
AI Hospital Management System - Main Application
Modular and organized architecture
"""
import datetime
import pandas as pd
import streamlit as st

# Import modular components
from modules.config import CHUTES_MODEL, DEFAULT_MEDICINES
from modules.ui_components import (
    setup_page_config, initialize_session_state, 
    render_sidebar, render_main_header, 
    render_patient_form, display_analysis_results,
    safe_rerun
)
from modules.data_manager import (
    load_stock, save_stock, load_hospitals, save_hospitals
)
from modules.clinical_analysis import (
    calc_severity_score, analyze_with_chutes
)
from modules.admission import (
    render_admission_workflow, render_admission_timeline
)
from modules.visualizations import (
    create_occupancy_bar_chart, create_bed_availability_chart,
    create_grouped_occupancy_chart, create_occupancy_heatmap
)

# Setup
setup_page_config()
initialize_session_state()

# Render UI components
render_sidebar()
render_main_header()

# Patient form
form_data = render_patient_form()

if form_data['submitted']:
    # Calculate severity score
    features = {
        'pid': form_data['pid'],
        'age': form_data['age'],
        'sex': form_data['sex'],
        'bmi': form_data['bmi'],
        'blood_pressure_sys': form_data['blood_pressure_sys'],
        'blood_pressure_dia': form_data['blood_pressure_dia'],
        'heart_rate': form_data['heart_rate'],
        'temperature': form_data['temperature'],
        'symptom_cough': form_data['symptom_cough'],
        'symptom_fever': form_data['symptom_fever'],
        'symptom_breathless': form_data['symptom_breathless'],
        'symptom_chest_pain': form_data['symptom_chest_pain'],
        'symptom_neuro': form_data['symptom_neuro'],
        'comorbidity_diabetes': form_data['comorbidity_diabetes'],
        'comorbidity_hypertension': form_data['comorbidity_hypertension'],
        'lab_wbc': form_data['lab_wbc'],
        'lab_crp': form_data['lab_crp']
    }
    severity_score = calc_severity_score(features)
    features['severity_score'] = severity_score

    # Store patient data
    st.session_state.current_patient = {
        'pid': form_data['pid'],
        'features': features,
        'severity_score': severity_score,
        'timestamp': datetime.datetime.now()
    }

    # Analyze with AI
    ai_result = analyze_with_chutes(features)

    if ai_result:
        st.session_state.current_patient['ai_result'] = ai_result

    st.session_state.admission_complete = False
    safe_rerun()

# Display analysis results
if st.session_state.current_patient and not st.session_state.admission_complete:
    patient = st.session_state.current_patient
    ai_result = patient.get('ai_result', {})
    
    display_analysis_results(patient, ai_result)
    
    # Admission workflow for inpatients
    if ai_result:
        render_admission_workflow(patient, ai_result)

# Inventory management
st.markdown("---")
st.subheader("Medicine Inventory Management")

stock_df = load_stock()

col1, col2, col3 = st.columns(3)
with col1:
    if st.button('Refresh Inventory'):
        load_stock.clear()
        safe_rerun()

with col2:
    if st.button('Replenish All Stock (+50 units)'):
        stock_df['stock'] = stock_df['stock'] + 50
        save_stock(stock_df)
        st.success("All stock replenished by 50 units")
        load_stock.clear()
        safe_rerun()

with col3:
    if st.button('Reset to Default (50 units each)'):
        stock_df = pd.DataFrame({'medicine_name': DEFAULT_MEDICINES, 'stock': [50]*len(DEFAULT_MEDICINES)})
        save_stock(stock_df)
        st.success("Stock reset to defaults")
        load_stock.clear()
        safe_rerun()

st.dataframe(stock_df, use_container_width=True)

# Ward Capacity Monitor
st.markdown("---")
st.subheader("Ward Capacity Monitor & Auto-Routing Status")

hospitals_df = load_hospitals()

# Prepare ward summary data
ward_summary = []
for ward_type in ['General', 'ICU', 'Neurological']:
    df = hospitals_df.copy()
    df['total_beds'] = pd.to_numeric(df['total_beds'], errors='coerce').fillna(0).astype(int)
    df['occupied_beds'] = pd.to_numeric(df['occupied_beds'], errors='coerce').fillna(0).astype(int)
    
    matching = df[df['ward_type'].str.lower() == ward_type.lower()]
    if matching.empty:
        continue
    
    total_beds = int(matching['total_beds'].sum())
    total_occupied = int(matching['occupied_beds'].sum())
    total_available = total_beds - total_occupied
    occupancy_pct = int((total_occupied / total_beds) * 100) if total_beds > 0 else 0
    
    ward_summary.append({
        'ward_type': ward_type,
        'total_beds': total_beds,
        'occupied_beds': total_occupied,
        'available_beds': total_available,
        'occupancy_pct': occupancy_pct
    })

# Create tabs for different views
tab1, tab2 = st.tabs(["Overall Summary", "Per-Hospital Breakdown"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_occupancy_bar_chart(ward_summary), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_bed_availability_chart(ward_summary), use_container_width=True)

with tab2:
    # Hospital breakdown
    hospital_breakdown = []
    for _, row in hospitals_df.iterrows():
        hospital_breakdown.append({
            'Hospital': row['hospital_name'],
            'Ward': row['ward_type'],
            'Total Beds': int(row['total_beds']),
            'Occupied': int(row['occupied_beds']),
            'Available': int(row['total_beds']) - int(row['occupied_beds']),
            'Occupancy %': int((int(row['occupied_beds']) / int(row['total_beds']) * 100) if int(row['total_beds']) > 0 else 0)
        })
    
    breakdown_df = pd.DataFrame(hospital_breakdown)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_grouped_occupancy_chart(breakdown_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_occupancy_heatmap(breakdown_df), use_container_width=True)
    
    st.write("**Detailed Breakdown:**")
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

# Summary statistics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

total_beds_all = sum([item['total_beds'] for item in ward_summary])
total_occupied_all = sum([item['occupied_beds'] for item in ward_summary])
total_available_all = sum([item['available_beds'] for item in ward_summary])
avg_occupancy = int((total_occupied_all / total_beds_all * 100) if total_beds_all > 0 else 0)

with col1:
    st.metric("Total Beds", total_beds_all)

with col2:
    st.metric("Occupied Beds", total_occupied_all)

with col3:
    st.metric("Available Beds", total_available_all)

with col4:
    st.metric("Average Occupancy", f"{avg_occupancy}%")

# Hospital management
st.markdown("---")
st.subheader("Hospital Bed Management")

st.dataframe(hospitals_df, use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button('Refresh Hospital Status'):
        load_hospitals.clear()
        safe_rerun()

with col2:
    if st.button('Reset All Occupancy (Clear Beds)'):
        hospitals_df['occupied_beds'] = 0
        save_hospitals(hospitals_df)
        st.success("All beds cleared")
        safe_rerun()

with col3:
    if st.button('Simulate Admissions (+2 per ward)'):
        hospitals_df['occupied_beds'] = (hospitals_df['occupied_beds'] + 2).clip(upper=hospitals_df['total_beds'])
        save_hospitals(hospitals_df)
        st.info("Simulated +2 admissions per ward")
        safe_rerun()


# Admission Timeline
render_admission_timeline()
