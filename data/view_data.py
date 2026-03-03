import pandas as pd

# Load data
df = pd.read_csv('data/synthetic_ehr_full.csv')

# Show first patient's timeline
patient_1 = df[df['patient_id'] == 'P0001']

print("Patient P0001 Timeline:")
print("="*80)
print(patient_1[['record_date', 'weight_kg', 'heart_rate_bpm', 'systolic_bp_mmhg', 
                 'diastolic_bp_mmhg', 'temperature_celsius', 'has_error', 'error_type']])

print("\n\nError Examples:")
print("="*80)
errors = df[df['has_error'] == True].head(5)
print(errors[['patient_id', 'record_date', 'weight_kg', 'heart_rate_bpm', 
              'error_type']])