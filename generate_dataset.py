
import numpy as np
import pandas as pd
import argparse

def gen_row(i):
    age = int(np.random.choice(range(0, 100)))
    sex = np.random.choice(["M","F"])
    bmi = round(np.random.normal(24,4),1)
    bp_sys = int(np.random.normal(120,15))
    bp_dia = int(np.random.normal(80,10))
    hr = int(np.random.normal(78,12))
    temp = round(np.random.normal(36.7,0.6),1)
    cough = np.random.binomial(1,0.25)
    fever = np.random.binomial(1,0.20)
    breathless = np.random.binomial(1,0.10)
    diabetes = np.random.binomial(1,0.12)
    hypertension = np.random.binomial(1,0.20)
    wbc = round(np.random.normal(7,2),1)
    crp = round(abs(np.random.normal(5,4)),1)
    severity = (fever*1 + breathless*2 + (wbc>11)*1 + (crp>10)*1 + (age>65)*1 + diabetes*1 + hypertension*1)
    p_inpatient = min(0.05 + severity*0.12, 0.95)
    inpatient = int(np.random.rand() < p_inpatient)
    ward = np.random.choice(["ICU","General","Isolation","Maternity"]) if inpatient else "None"
    stay = int(max(1, np.random.poisson(3) + (ward=="ICU")*4)) if inpatient else 0
    diagnosis_code = np.random.choice(["D01","D02","D03","D04","D05","D06"])
    return {
        "patient_id": f"P{i:06d}",
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "blood_pressure_sys": bp_sys,
        "blood_pressure_dia": bp_dia,
        "heart_rate": hr,
        "temperature": temp,
        "symptom_cough": cough,
        "symptom_fever": fever,
        "symptom_breathless": breathless,
        "comorbidity_diabetes": diabetes,
        "comorbidity_hypertension": hypertension,
        "lab_wbc": wbc,
        "lab_crp": crp,
        "diagnosis_code": diagnosis_code,
        "severity_score": severity,
        "inpatient": inpatient,
        "ward_type": ward,
        "stay_days": stay
    }

def generate(n, out):
    rows = [gen_row(i) for i in range(1, n+1)]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} rows to {out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000, help='number of samples')
    parser.add_argument('--out', type=str, default='data/patients_sample.csv', help='output csv path')
    args = parser.parse_args()
    generate(args.n, args.out)
