import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

class AnomalyDetector:
    """
    Agent 3: Statistical anomaly detection across population
    Uses Z-score and IQR methods to identify outliers
    """
    
    def __init__(self):
        self.name = "Anomaly Detector"
        
        # Fields to analyze
        self.vital_fields = [
            'weight_kg',
            'heart_rate_bpm',
            'systolic_bp_mmhg',
            'diastolic_bp_mmhg',
            'temperature_celsius'
        ]
        
        # Population statistics (will be calculated from all data)
        self.population_stats = None
    
    def fit_population_statistics(self, all_data):
        """
        Calculate population-wide statistics from all patients
        
        Parameters:
        - all_data: DataFrame with ALL patients' data
        """
        self.population_stats = {}
        
        for field in self.vital_fields:
            values = all_data[field].values
            
            # Remove extreme outliers first for better statistics
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            # Filter outliers for clean stats
            clean_values = values[(values >= q1 - 3*iqr) & (values <= q3 + 3*iqr)]
            
            self.population_stats[field] = {
                'mean': np.mean(clean_values),
                'std': np.std(clean_values),
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'min': np.min(clean_values),
                'max': np.max(clean_values)
            }
        
        print(f"✓ Population statistics calculated from {len(all_data)} records")
    
    def validate_patient_timeline(self, patient_data):
        """
        Detect anomalies in patient records
        
        Parameters:
        - patient_data: DataFrame with one patient's timeline
        
        Returns:
        - List of validation results
        """
        if self.population_stats is None:
            raise ValueError("Must call fit_population_statistics() first!")
        
        results = []
        
        for idx, record in patient_data.iterrows():
            anomalies = self._detect_anomalies(record)
            
            valid = len(anomalies) == 0
            
            # Confidence based on severity
            if valid:
                confidence = 0.85
            else:
                # Higher Z-score = higher confidence it's anomaly
                max_z = max([a['z_score'] for a in anomalies])
                confidence = min(0.7 + (max_z * 0.05), 0.95)
            
            explanation = self._generate_explanation(record, anomalies)
            
            results.append({
                'record_date': record['record_date'],
                'patient_id': record['patient_id'],
                'valid': valid,
                'confidence': round(confidence, 2),
                'anomalies': anomalies,
                'explanation': explanation
            })
        
        return results
    
    def _detect_anomalies(self, record):
        """Detect statistical anomalies using Z-score and IQR methods"""
        anomalies = []
        
        for field in self.vital_fields:
            value = record[field]
            stats = self.population_stats[field]
            
            # Method 1: Z-score (how many std devs from mean)
            z_score = abs((value - stats['mean']) / stats['std']) if stats['std'] > 0 else 0
            
            # Method 2: IQR outlier detection
            lower_bound = stats['q1'] - 1.5 * stats['iqr']
            upper_bound = stats['q3'] + 1.5 * stats['iqr']
            is_iqr_outlier = value < lower_bound or value > upper_bound
            
            # Flag if either method detects anomaly
            # Z-score > 4 OR outside IQR bounds
            if z_score > 4 or is_iqr_outlier:
                anomaly_type = []
                if z_score > 4:
                    anomaly_type.append(f"Z-score: {round(z_score, 2)}")
                if is_iqr_outlier:
                    if value < lower_bound:
                        anomaly_type.append(f"Below IQR bound ({round(lower_bound, 2)})")
                    else:
                        anomaly_type.append(f"Above IQR bound ({round(upper_bound, 2)})")
                
                anomalies.append({
                    'field': field,
                    'value': value,
                    'z_score': round(z_score, 2),
                    'population_mean': round(stats['mean'], 2),
                    'population_std': round(stats['std'], 2),
                    'iqr_bounds': (round(lower_bound, 2), round(upper_bound, 2)),
                    'anomaly_type': ', '.join(anomaly_type),
                    'severity': 'HIGH' if z_score > 6 else 'MEDIUM'
                })
        
        return anomalies
    
    def _generate_explanation(self, record, anomalies):
        """Generate simple explanation without LLM"""
        if not anomalies:
            return "All values within normal population range"
        
        explanations = []
        for anomaly in anomalies:
            field_name = anomaly['field'].replace('_', ' ').title()
            exp = f"{field_name}: {anomaly['value']} is statistical outlier ({anomaly['anomaly_type']})"
            explanations.append(exp)
        
        severity = "CRITICAL" if any(a['severity'] == 'HIGH' for a in anomalies) else "WARNING"
        
        return f"[{severity}] " + "; ".join(explanations) + f". Population mean ± std: {anomalies[0]['field']} = {anomalies[0]['population_mean']}±{anomalies[0]['population_std']}"

# Test Agent 3
if __name__ == "__main__":
    # Load ALL data to calculate population statistics
    df_all = pd.read_csv('data/synthetic_ehr_test.csv')
    
    agent = AnomalyDetector()
    print(f"Testing {agent.name}...")
    print("="*80)
    
    # First, fit population statistics
    agent.fit_population_statistics(df_all)
    
    # Test on first patient
    patient_1 = df_all[df_all['patient_id'] == 'P0001']
    
    results = agent.validate_patient_timeline(patient_1)
    
    # Show results
    for result in results:
        status = "✓ VALID" if result['valid'] else "✗ FLAGGED"
        print(f"\n{result['record_date']}: {status} (confidence: {result['confidence']})")
        print(f"  {result['explanation']}")
        
        if result['anomalies']:
            print(f"  Anomalies: {len(result['anomalies'])}")
            for anomaly in result['anomalies']:
                print(f"    - {anomaly['field']}: {anomaly['value']} (Z={anomaly['z_score']}, severity: {anomaly['severity']})")