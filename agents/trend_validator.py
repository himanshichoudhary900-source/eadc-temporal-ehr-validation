import google.generativeai as genai
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class TrendValidator:
    """
    Agent 1: Validates realistic temporal trends
    Checks if changes between consecutive records are physiologically plausible
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.name = "Trend Validator"
        
        # Define maximum realistic changes per time period (15 days)
        self.max_realistic_changes = {
            'weight_kg': 5,          # Max 5kg change in 15 days
            'heart_rate_bpm': 20,    # Max 20 bpm change
            'systolic_bp_mmhg': 20,  # Max 20 mmHg change
            'diastolic_bp_mmhg': 15, # Max 15 mmHg change
            'temperature_celsius': 2 # Max 2°C change
        }
    
    def validate_patient_timeline(self, patient_data):
        """
        Validate all records for one patient
        
        Parameters:
        - patient_data: DataFrame with one patient's timeline (sorted by date)
        
        Returns:
        - List of validation results for each record
        """
        results = []
        
        # Sort by date
        patient_data = patient_data.sort_values('record_date')
        
        for idx in range(len(patient_data)):
            current_record = patient_data.iloc[idx]
            
            # First record has no previous record to compare
            if idx == 0:
                results.append({
                    'record_date': current_record['record_date'],
                    'patient_id': current_record['patient_id'],
                    'valid': True,
                    'confidence': 1.0,
                    'issues': [],
                    'explanation': "First record in timeline - no comparison possible"
                })
                continue
            
            # Compare with previous record
            previous_record = patient_data.iloc[idx - 1]
            issues = self._check_trends(previous_record, current_record)
            
            # Determine validity
            valid = len(issues) == 0
            confidence = 1.0 if valid else 0.9  # High confidence when flagging
            
            # Generate explanation
            if valid:
                explanation = f"All changes within realistic limits compared to {previous_record['record_date']}"
            else:
                explanation = self._generate_explanation(previous_record, current_record, issues)
            
            results.append({
                'record_date': current_record['record_date'],
                'patient_id': current_record['patient_id'],
                'valid': valid,
                'confidence': confidence,
                'issues': issues,
                'explanation': explanation
            })
        
        return results
    
    def _check_trends(self, prev_record, curr_record):
        """Check if changes between records are realistic"""
        issues = []
        
        # Check each vital sign
        for field, max_change in self.max_realistic_changes.items():
            prev_val = prev_record[field]
            curr_val = curr_record[field]
            
            delta = abs(curr_val - prev_val)
            
            if delta > max_change:
                issues.append({
                    'field': field,
                    'previous_value': prev_val,
                    'current_value': curr_val,
                    'delta': delta,
                    'max_realistic': max_change,
                    'severity': 'HIGH' if delta > max_change * 2 else 'MEDIUM'
                })
        
        return issues
    
    def _generate_explanation(self, prev_record, curr_record, issues):
        """Generate human-readable explanation using LLM"""
        
        # Build context for LLM
        context = f"""
        Previous record ({prev_record['record_date']}):
        - Weight: {prev_record['weight_kg']}kg
        - Heart Rate: {prev_record['heart_rate_bpm']}bpm
        - BP: {prev_record['systolic_bp_mmhg']}/{prev_record['diastolic_bp_mmhg']}mmHg
        - Temperature: {prev_record['temperature_celsius']}°C
        
        Current record ({curr_record['record_date']}):
        - Weight: {curr_record['weight_kg']}kg
        - Heart Rate: {curr_record['heart_rate_bpm']}bpm
        - BP: {curr_record['systolic_bp_mmhg']}/{curr_record['diastolic_bp_mmhg']}mmHg
        - Temperature: {curr_record['temperature_celsius']}°C
        
        Issues detected:
        """
        
        for issue in issues:
            context += f"\n- {issue['field']}: Changed from {issue['previous_value']} to {issue['current_value']} (Δ={issue['delta']}, Max realistic={issue['max_realistic']})"
        
        prompt = f"""
        You are a medical data validator. Analyze the temporal trend issue:
        
        {context}
        
        Provide a brief explanation (2-3 sentences):
        1. What changed unrealistically?
        2. Why is this concerning?
        3. Most likely cause (data entry error, equipment malfunction, etc.)
        
        Be concise and professional.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            # Fallback if LLM fails
            return f"Unrealistic changes detected in {len(issues)} field(s). Manual review recommended."

# Test the agent
if __name__ == "__main__":
    # Load test data
    df = pd.read_csv('data/synthetic_ehr_test.csv')
    
    # Test on first patient
    patient_1 = df[df['patient_id'] == 'P0001']
    
    agent = TrendValidator()
    print(f"Testing {agent.name}...")
    print("="*80)
    
    results = agent.validate_patient_timeline(patient_1)
    
    # Show results
    for result in results:
        status = "✓ VALID" if result['valid'] else "✗ FLAGGED"
        print(f"\n{result['record_date']}: {status} (confidence: {result['confidence']})")
        print(f"  {result['explanation']}")
        
        if result['issues']:
            print(f"  Issues: {len(result['issues'])}")
            for issue in result['issues']:
                print(f"    - {issue['field']}: {issue['previous_value']} → {issue['current_value']} (Δ{issue['delta']})")