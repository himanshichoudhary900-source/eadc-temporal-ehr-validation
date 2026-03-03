import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def assign_provenance(record_num, has_error):
    """
    Assign data provenance (source) to each record
    More errors come from manual entry, fewer from sensors
    """
    
    # Define source types with probabilities
    if has_error:
        # Errors more likely from manual entry
        source_choices = ['manual_entry', 'sensor', 'lab_verified']
        source_weights = [0.6, 0.3, 0.1]  # 60% manual, 30% sensor, 10% lab
    else:
        # Clean data more likely from automated sources
        source_choices = ['manual_entry', 'sensor', 'lab_verified']
        source_weights = [0.3, 0.5, 0.2]  # 30% manual, 50% sensor, 20% lab
    
    source = random.choices(source_choices, weights=source_weights)[0]
    
    # Assign device/entry details
    if source == 'manual_entry':
        device_id = f"NURSE_{random.randint(1, 10):02d}"
        entry_method = "keyboard"
    elif source == 'sensor':
        device_id = f"DEVICE_{random.randint(100, 150):03d}"
        entry_method = "automated"
    else:  # lab_verified
        device_id = f"LAB_{random.randint(1, 5):02d}"
        entry_method = "verified"
    
    return {
        'source': source,
        'device_id': device_id,
        'entry_method': entry_method,
        'source_reliability': {
            'manual_entry': 0.75,
            'sensor': 0.90,
            'lab_verified': 0.95
        }[source]
    }


def generate_synthetic_ehr_data(n_patients=100, records_per_patient=10, error_rate=0.30):
    """
    Generate synthetic EHR temporal data with intentional errors AND provenance
    
    Parameters:
    - n_patients: Number of patients
    - records_per_patient: Timeline length per patient
    - error_rate: Percentage of records with errors (0.30 = 30%)
    """
    
    np.random.seed(42)
    random.seed(42)
    
    all_data = []
    
    for patient_id in range(1, n_patients + 1):
        # Generate patient baseline (normal values for this patient)
        baseline_weight = np.random.randint(50, 90)  # kg
        baseline_hr = np.random.randint(65, 85)      # bpm
        baseline_sys_bp = np.random.randint(110, 130) # mmHg
        baseline_dia_bp = np.random.randint(70, 85)   # mmHg
        baseline_temp = round(np.random.uniform(36.2, 36.8), 1)  # Celsius
        
        start_date = datetime(2024, 1, 1)
        
        for record_num in range(records_per_patient):
            # Calculate date (every 15 days)
            record_date = start_date + timedelta(days=record_num * 15)
            
            # Decide if this record will have an error
            has_error = random.random() < error_rate
            
            if not has_error:
                # NORMAL VALUES (small natural variation)
                weight = baseline_weight + np.random.randint(-2, 3)
                heart_rate = baseline_hr + np.random.randint(-5, 6)
                systolic_bp = baseline_sys_bp + np.random.randint(-5, 6)
                diastolic_bp = baseline_dia_bp + np.random.randint(-3, 4)
                temperature = round(baseline_temp + np.random.uniform(-0.3, 0.3), 1)
                error_type = None
                
            else:
                # INJECT ERROR
                error_type = random.choice([
                    'typo_weight',
                    'typo_hr',
                    'impossible_hr',
                    'bp_swap',
                    'temp_unit_error',
                    'extreme_jump'
                ])
                
                if error_type == 'typo_weight':
                    # Extra digit: 70 → 700
                    weight = baseline_weight * 10
                    heart_rate = baseline_hr + np.random.randint(-5, 6)
                    systolic_bp = baseline_sys_bp + np.random.randint(-5, 6)
                    diastolic_bp = baseline_dia_bp + np.random.randint(-3, 4)
                    temperature = round(baseline_temp + np.random.uniform(-0.3, 0.3), 1)
                    
                elif error_type == 'typo_hr':
                    # Typo: 75 → 750
                    weight = baseline_weight + np.random.randint(-2, 3)
                    heart_rate = baseline_hr * 10
                    systolic_bp = baseline_sys_bp + np.random.randint(-5, 6)
                    diastolic_bp = baseline_dia_bp + np.random.randint(-3, 4)
                    temperature = round(baseline_temp + np.random.uniform(-0.3, 0.3), 1)
                    
                elif error_type == 'impossible_hr':
                    # Physiologically impossible
                    weight = baseline_weight + np.random.randint(-2, 3)
                    heart_rate = random.choice([0, -10, 300, 500])
                    systolic_bp = baseline_sys_bp + np.random.randint(-5, 6)
                    diastolic_bp = baseline_dia_bp + np.random.randint(-3, 4)
                    temperature = round(baseline_temp + np.random.uniform(-0.3, 0.3), 1)
                    
                elif error_type == 'bp_swap':
                    # Systolic and diastolic swapped
                    weight = baseline_weight + np.random.randint(-2, 3)
                    heart_rate = baseline_hr + np.random.randint(-5, 6)
                    systolic_bp = baseline_dia_bp  # WRONG
                    diastolic_bp = baseline_sys_bp  # WRONG
                    temperature = round(baseline_temp + np.random.uniform(-0.3, 0.3), 1)
                    
                elif error_type == 'temp_unit_error':
                    # Fahrenheit entered instead of Celsius
                    weight = baseline_weight + np.random.randint(-2, 3)
                    heart_rate = baseline_hr + np.random.randint(-5, 6)
                    systolic_bp = baseline_sys_bp + np.random.randint(-5, 6)
                    diastolic_bp = baseline_dia_bp + np.random.randint(-3, 4)
                    temperature = round(baseline_temp * 1.8 + 32, 1)  # Convert to F
                    
                else:  # extreme_jump
                    # Unrealistic sudden change
                    weight = baseline_weight + random.choice([-30, 30, 50])
                    heart_rate = baseline_hr + random.choice([-40, 60, 80])
                    systolic_bp = baseline_sys_bp + random.choice([-50, 70])
                    diastolic_bp = baseline_dia_bp + random.choice([-30, 40])
                    temperature = round(baseline_temp + random.choice([-2, 3, 4]), 1)
            
            # ===== ADD PROVENANCE (NEW) =====
            provenance = assign_provenance(record_num, has_error)
            
            # Create record WITH provenance
            record = {
                'patient_id': f'P{patient_id:04d}',
                'record_date': record_date.strftime('%Y-%m-%d'),
                'weight_kg': weight,
                'heart_rate_bpm': heart_rate,
                'systolic_bp_mmhg': systolic_bp,
                'diastolic_bp_mmhg': diastolic_bp,
                'temperature_celsius': temperature,
                
                # Ground truth (for evaluation)
                'has_error': has_error,
                'error_type': error_type,
                'baseline_weight': baseline_weight,
                'baseline_hr': baseline_hr,
                
                # PROVENANCE FIELDS (NEW)
                'source': provenance['source'],
                'device_id': provenance['device_id'],
                'entry_method': provenance['entry_method'],
                'source_reliability': provenance['source_reliability']
            }
            
            all_data.append(record)
    
    df = pd.DataFrame(all_data)
    return df


# Generate and save
if __name__ == "__main__":
    print("Generating synthetic EHR temporal data with provenance...")
    
    df = generate_synthetic_ehr_data(
        n_patients=100,
        records_per_patient=10,
        error_rate=0.30
    )
    
    # Save full dataset with labels AND provenance
    df.to_csv('data/synthetic_ehr_full_provenance.csv', index=False)
    print(f"✓ Generated {len(df)} records for {df['patient_id'].nunique()} patients")
    
    # Save clean version (without ground truth labels)
    df_clean = df.drop(columns=['has_error', 'error_type', 'baseline_weight', 'baseline_hr'])
    df_clean.to_csv('data/synthetic_ehr_test_provenance.csv', index=False)
    print(f"✓ Saved test dataset with provenance (without labels)")
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Total records: {len(df)}")
    print(f"Records with errors: {df['has_error'].sum()} ({df['has_error'].mean()*100:.1f}%)")
    print(f"\nError types:")
    print(df[df['has_error']]['error_type'].value_counts())
    
    # Print provenance statistics
    print(f"\n{'='*60}")
    print("PROVENANCE STATISTICS")
    print(f"{'='*60}")
    print("\nData source distribution:")
    print(df['source'].value_counts())
    print(f"\nError rate by source:")
    error_by_source = df.groupby('source')['has_error'].agg(['sum', 'count', 'mean'])
    error_by_source.columns = ['Errors', 'Total', 'Error Rate']
    print(error_by_source)
    print(f"\nAverage source reliability:")
    print(df.groupby('source')['source_reliability'].first())
    
    print("\n✓ Data generation with provenance complete!")