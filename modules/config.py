"""
Configuration and Constants
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# Chutes AI configuration
CHUTES_MODEL = os.getenv("CHUTES_MODEL", "unsloth/gemma-3-12b-it")
CHUTES_API_URL = os.getenv("CHUTES_API_URL", "https://llm.chutes.ai/v1/chat/completions")
CHUTES_API_TOKEN = os.getenv("CHUTES_API_TOKEN")
CHUTES_TIMEOUT = int(os.getenv("CHUTES_TIMEOUT", "30"))

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
STOCK_CSV = os.path.join(DATA_DIR, 'medicine_stock.csv')
HOSPITALS_CSV = os.path.join(DATA_DIR, 'hospitals.csv')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Disease categories for reference
HOSPITALIZATION_DISEASES = {
    "J18.9": "Pneumonia, unspecified",
    "I21.9": "Acute myocardial infarction",
    "I63.9": "Cerebral infarction",
    "A41.9": "Sepsis, unspecified",
    "J15.9": "Bacterial pneumonia"
}

OUTPATIENT_DISEASES = {
    "I10": "Essential hypertension",
    "E11.9": "Type 2 diabetes",
    "J06.9": "Acute upper respiratory infection",
    "K29.70": "Gastritis",
    "M54.50": "Low back pain"
}

HEALTHY_CODE = {
    "Z00.0": "General medical examination"
}

# Default medicine list
DEFAULT_MEDICINES = [
    'Amoxicillin 500mg', 'Azithromycin 250mg', 'Paracetamol 500mg',
    'Aspirin 100mg', 'Clopidogrel 75mg', 'Atorvastatin 20mg',
    'Meropenem 1g IV', 'Vancomycin 1g IV', 'IV Fluids',
    'Ceftriaxone 1g IV', 'Oxygen therapy', 'Mannitol IV',
    'Amlodipine 5mg', 'Lisinopril 10mg', 'Metformin 500mg',
    'Omeprazole 20mg', 'Ibuprofen 400mg', 'Chlorpheniramine 4mg'
]

# Default hospital data
DEFAULT_HOSPITALS = [
    {'hospital_id': 'H-01', 'hospital_name': 'Central Medical Hospital', 'ward_type': 'General', 'total_beds': 20, 'occupied_beds': 15},
    {'hospital_id': 'H-02', 'hospital_name': 'Central Medical Hospital', 'ward_type': 'ICU', 'total_beds': 6, 'occupied_beds': 4},
    {'hospital_id': 'H-03', 'hospital_name': 'Central Medical Hospital', 'ward_type': 'Neurological', 'total_beds': 8, 'occupied_beds': 3},
    {'hospital_id': 'H-04', 'hospital_name': 'City General Hospital', 'ward_type': 'General', 'total_beds': 25, 'occupied_beds': 8},
    {'hospital_id': 'H-05', 'hospital_name': 'City General Hospital', 'ward_type': 'ICU', 'total_beds': 8, 'occupied_beds': 2},
    {'hospital_id': 'H-06', 'hospital_name': 'City General Hospital', 'ward_type': 'Neurological', 'total_beds': 10, 'occupied_beds': 5},
]
