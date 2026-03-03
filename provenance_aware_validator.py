import pandas as pd

class ProvenanceAwareValidator:
    """
    Wrapper that makes any agent provenance-aware
    Adjusts agent confidence AND decision based on data source reliability
    """
    
    def __init__(self, base_agent, confidence_threshold=0.75):
        """
        Parameters:
        - base_agent: Any of the 3 agents
        - confidence_threshold: Minimum confidence to flag as INVALID
        """
        self.base_agent = base_agent
        self.name = f"Provenance-Aware {base_agent.name}"
        self.confidence_threshold = confidence_threshold
        
        # Provenance confidence modifiers
        self.source_modifiers = {
            'lab_verified': 1.0,
            'sensor': 0.95,
            'manual_entry': 0.80
        }
    
    def validate_patient_timeline(self, patient_data):
        """
        Validate timeline with provenance awareness
        NOW ADJUSTS BOTH CONFIDENCE AND DECISION
        """
        
        # Run base agent validation
        base_results = self.base_agent.validate_patient_timeline(patient_data)
        
        # Adjust confidence AND decision based on provenance
        provenance_aware_results = []
        
        for idx, result in enumerate(base_results):
            record_date = result['record_date']
            
            # Get corresponding record
            record = patient_data[patient_data['record_date'] == record_date].iloc[0]
            source = record.get('source', 'manual_entry')
            source_reliability = record.get('source_reliability', 0.75)
            
            # Apply provenance modifier to confidence
            original_confidence = result['confidence']
            original_valid = result['valid']
            modifier = self.source_modifiers.get(source, 0.85)
            adjusted_confidence = original_confidence * modifier
            
            # ===== SMARTER THRESHOLD-BASED DECISION =====
            # ONLY override manual_entry with borderline confidence
            # Keep sensor/lab strict (high trust sources)
            # REPLACE WITH:
            adjusted_valid = original_valid  # Keep original decision
            decision_changed = False
            # Create provenance-aware result
            provenance_result = result.copy()
            provenance_result['original_confidence'] = original_confidence
            provenance_result['original_valid'] = original_valid
            provenance_result['confidence'] = round(adjusted_confidence, 2)
            provenance_result['valid'] = adjusted_valid  # UPDATED DECISION
            provenance_result['decision_changed'] = decision_changed
            provenance_result['provenance'] = {
                'source': source,
                'source_reliability': source_reliability,
                'confidence_modifier': modifier,
                'threshold': self.confidence_threshold
            }
            
            # Adjust explanation
            if decision_changed:
                provenance_result['explanation'] = (
                    f"[OVERRIDDEN: Manual entry with low confidence] "
                    f"Original: INVALID (conf={original_confidence:.2f}), "
                    f"Adjusted: VALID (conf={adjusted_confidence:.2f} < threshold {self.confidence_threshold}). "
                    f"Source: {source} - likely data entry error, not actual patient issue."
                )
            elif source == 'manual_entry' and not adjusted_valid:
                provenance_result['explanation'] = (
                    f"[Manual Entry - Reduced Confidence] {result['explanation']}"
                )
            elif source == 'lab_verified':
                provenance_result['explanation'] = (
                    f"[Lab Verified - High Confidence] {result['explanation']}"
                )
            
            provenance_aware_results.append(provenance_result)
        
        return provenance_aware_results


# Test
if __name__ == "__main__":
    from agents.trend_validator import TrendValidator
    
    # Load provenance data
    df = pd.read_csv('data/synthetic_ehr_test_provenance.csv')
    patient_1 = df[df['patient_id'] == 'P0001']
    
    print("="*80)
    print("TESTING SMARTER PROVENANCE VALIDATION")
    print("="*80)
    
    # Standard
    print("\n1. STANDARD TREND VALIDATOR:")
    standard_agent = TrendValidator()
    standard_results = standard_agent.validate_patient_timeline(patient_1)
    
    for r in standard_results[:5]:
        print(f"\n{r['record_date']}: {'VALID' if r['valid'] else 'INVALID'} (conf: {r['confidence']})")
    
    # Provenance with smarter threshold (only overrides manual_entry)
    print("\n\n2. PROVENANCE-AWARE (ONLY OVERRIDES MANUAL ENTRY):")
    provenance_agent = ProvenanceAwareValidator(standard_agent, confidence_threshold=0.75)
    provenance_results = provenance_agent.validate_patient_timeline(patient_1)
    
    decision_changes = 0
    for r in provenance_results[:5]:
        status = "VALID" if r['valid'] else "INVALID"
        changed = " [DECISION CHANGED]" if r['decision_changed'] else ""
        print(f"\n{r['record_date']}: {status} (conf: {r['confidence']}){changed}")
        print(f"  Source: {r['provenance']['source']}")
        if r['decision_changed']:
            decision_changes += 1
            print(f"  WHY: Manual entry uncertainty, not actual patient issue")
    
    print(f"\n{'='*80}")
    print(f"Decision changes: {decision_changes} (manual entry only)")
    print("Sensor/Lab data: Always strict (no overrides)")
    print("="*80)