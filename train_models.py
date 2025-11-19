
import pandas as pd
import argparse
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def build_and_save_models(data_path, out_dir):
    df = pd.read_csv(data_path)
    feature_cols = ['age','bmi','blood_pressure_sys','blood_pressure_dia','heart_rate','temperature',
                    'symptom_cough','symptom_fever','symptom_breathless','comorbidity_diabetes','comorbidity_hypertension',
                    'lab_wbc','lab_crp','diagnosis_code','severity_score']
    df = df.dropna(subset=feature_cols + ['inpatient','ward_type','stay_days'])
    X = df[feature_cols].copy()
    y_inp = df['inpatient']
    cat_features = ['diagnosis_code']
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)], remainder='passthrough')
    inp_pipeline = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
    inp_pipeline.fit(X, y_inp)
    joblib.dump(inp_pipeline, os.path.join(out_dir, 'inpatient_model.pkl'))
    print('Saved inpatient_model.pkl')
    df_in = df[df['inpatient']==1]
    if len(df_in) >= 50:
        Xw = df_in[feature_cols].copy()
        yw = df_in['ward_type']
        ward_pipeline = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
        ward_pipeline.fit(Xw, yw)
        joblib.dump(ward_pipeline, os.path.join(out_dir, 'ward_model.pkl'))
        print('Saved ward_model.pkl')
        Xs = df_in[feature_cols].copy()
        ys = df_in['stay_days']
        stay_pipeline = Pipeline([('pre', preprocessor), ('reg', RandomForestRegressor(n_estimators=100, random_state=42))])
        stay_pipeline.fit(Xs, ys)
        joblib.dump(stay_pipeline, os.path.join(out_dir, 'stay_model.pkl'))
        print('Saved stay_model.pkl')
    else:
        print('Not enough inpatient samples to train ward/stay models.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/patients_sample.csv')
    parser.add_argument('--out_dir', type=str, default='models')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    build_and_save_models(args.data, args.out_dir)
