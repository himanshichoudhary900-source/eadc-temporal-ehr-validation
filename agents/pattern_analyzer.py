import google.generativeai as genai
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class PatternAnalyzer:
    """
    Agent 2: Learns patient-specific patterns and detects deviations
    Uses statistical baseline (mean, std dev) from patient history
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.name = "Pattern Analyzer"
        
        # Fields to analyze
        self.vital_fields = [
            'weight_kg',
            'heart_rate_bpm',
            'systolic_bp_mmhg',
            'diastolic_bp_mmhg',
            'temperature_celsius'
        ]
    
    def validate_patient_timeline(self, patient_data):
        """
        Validate records based on patient's historical patterns
        
        Parameters:
        - patient_data: DataFrame with one patient's timeline
        
        Returns:
        - List of validation results
        """
        results = []
        
        # Sort by date
        patient_data = patient_data.sort_values('record_date').reset_index(drop=True)
        
        for idx in range(len(patient_data)):
            current_record = patient_data.iloc[idx]
            
            # Need at least 3 previous records to establish pattern
            if idx < 3:
                results.append({
                    'record_date': current_record['record_date'],
                    'patient_id': current_record['patient_id'],
                    'valid': True,
                    'confidence': 0.5,  # Low confidence - insufficient history
                    'deviations': [],
                    'explanation': f"Insufficient history ({idx} records) to establish baseline pattern"
                })
                continue
            
            # Use previous records to calculate baseline
            historical_data = patient_data.iloc[:idx]  # All records before current
            
            # Calculate baseline statistics
            baseline = self._calculate_baseline(historical_data)
            
            # Check for deviations
            deviations = self._check_deviations(current_record, baseline)
            
            # Determine validity
            valid = len(deviations) == 0
            
            # Confidence based on history length and deviation severity
            if valid:
                confidence = min(0.7 + (idx * 0.03), 0.95)  # More history = higher confidence
            else:
                # High deviation = high confidence it's wrong
                max_z_score = max([d['z_score'] for d in deviations]) if deviations else 0
                confidence = min(0.6 + (max_z_score * 0.1), 0.95)
            
            # Generate explanation
            explanation = self._generate_explanation(
                current_record, 
                baseline, 
                deviations, 
                len(historical_data)
            )
            
            results.append({
                'record_date': current_record['record_date'],
                'patient_id': current_record['patient_id'],
                'valid': valid,
                'confidence': round(confidence, 2),
                'deviations': deviations,
                'baseline': baseline,
                'explanation': explanation
            })
        
        return results
    
    def _calculate_baseline(self, historical_data):
        """Calculate statistical baseline from patient history"""
        baseline = {}
        
        for field in self.vital_fields:
            values = historical_data[field].values
            
            baseline[field] = {
                'mean': round(np.mean(values), 2),
                'std': round(np.std(values), 2),
                'min': round(np.min(values), 2),
                'max': round(np.max(values), 2),
                'count': len(values)
            }
        
        return baseline
    
    def _check_deviations(self, current_record, baseline):
        """Check if current values deviate significantly from baseline"""
        deviations = []
        
        for field in self.vital_fields:
            current_value = current_record[field]
            mean = baseline[field]['mean']
            std = baseline[field]['std']
            
            # Skip if std is 0 (no variation in history)
            if std == 0:
                continue
            
            # Calculate Z-score
            z_score = abs((current_value - mean) / std)
            
            # Flag if more than 3 standard deviations away
            if z_score > 3:
                deviations.append({
                    'field': field,
                    'current_value': current_value,
                    'baseline_mean': mean,
                    'baseline_std': std,
                    'z_score': round(z_score, 2),
                    'severity': 'HIGH' if z_score > 5 else 'MEDIUM'
                })
        
        return deviations
    
    def _generate_explanation(self, current_record, baseline, deviations, history_count):
        """Generate explanation using LLM"""
        
        if not deviations:
            return f"All values within expected range based on {history_count} historical records"
        
        # Build context
        context = f"""
        Patient baseline (from {history_count} historical records):
        """
        
        for field in self.vital_fields:
            b = baseline[field]
            context += f"\n- {field}: {b['mean']}±{b['std']} (range: {b['min']}-{b['max']})"
        
        context += f"\n\nCurrent record ({current_record['record_date']}):"
        for field in self.vital_fields:
            context += f"\n- {field}: {current_record[field]}"
        
        context += "\n\nDeviations detected:"
        for dev in deviations:
            context += f"\n- {dev['field']}: {dev['current_value']} (baseline: {dev['baseline_mean']}±{dev['baseline_std']}, Z-score: {dev['z_score']})"
        
        prompt = f"""
        You are analyzing patient vital signs patterns. 
        
        {context}
        
        Provide a brief explanation (2-3 sentences):
        1. Which value(s) deviate from this patient's normal pattern?
        2. How significant is the deviation?
        3. Possible causes (data error, actual medical change, etc.)
        
        Be concise and clinical.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Significant deviation from patient baseline in {len(deviations)} parameter(s). Z-scores: {[d['z_score'] for d in deviations]}"

# Test Agent 2
if __name__ == "__main__":
    # Load test data
    df = pd.read_csv('data/synthetic_ehr_test.csv')
    
    # Test on first patient
    patient_1 = df[df['patient_id'] == 'P0001']
    
    agent = PatternAnalyzer()
    print(f"Testing {agent.name}...")
    print("="*80)
    
    results = agent.validate_patient_timeline(patient_1)
    
    # Show results
    for result in results:
        status = "✓ VALID" if result['valid'] else "✗ FLAGGED"
        print(f"\n{result['record_date']}: {status} (confidence: {result['confidence']})")
        print(f"  {result['explanation']}")
        
        if result['deviations']:
            print(f"  Deviations: {len(result['deviations'])}")
            for dev in result['deviations']:
                print(f"    - {dev['field']}: {dev['current_value']} (baseline: {dev['baseline_mean']}±{dev['baseline_std']}, Z={dev['z_score']})")