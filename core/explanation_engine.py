from typing import Dict, List
import json

class ExplanationEngine:
    """
    Generates multi-layer explanations for validation decisions
    Layer 1: Summary
    Layer 2: Reasoning
    Layer 3: Evidence & Context
    """
    
    def __init__(self):
        self.name = "Explanation Engine"
    
    def generate_explanation(self, consensus_result: Dict, agent_results: Dict, 
                            record: Dict) -> Dict:
        """
        Generate complete multi-layer explanation
        
        Parameters:
        - consensus_result: Output from consensus engine
        - agent_results: Results from all agents
        - record: The data record being validated
        
        Returns:
        - Multi-layer explanation dictionary
        """
        
        # Layer 1: Summary (one sentence)
        layer1 = self._generate_layer1_summary(consensus_result, record)
        
        # Layer 2: Reasoning (why this decision)
        layer2 = self._generate_layer2_reasoning(consensus_result, agent_results)
        
        # Layer 3: Evidence & Context (detailed breakdown)
        layer3 = self._generate_layer3_evidence(consensus_result, agent_results, record)
        
        # Recommendation
        recommendation = self._generate_recommendation(consensus_result, agent_results)
        
        return {
            'layer1_summary': layer1,
            'layer2_reasoning': layer2,
            'layer3_evidence': layer3,
            'recommendation': recommendation,
            'full_explanation': self._format_full_explanation(layer1, layer2, layer3, recommendation)
        }
    
    def _generate_layer1_summary(self, consensus: Dict, record: Dict) -> str:
        """Layer 1: One-sentence summary"""
        
        decision = consensus['final_decision']
        confidence = consensus['confidence']
        date = record['record_date']
        patient = record['patient_id']
        
        if decision == 'INVALID':
            return f"❌ Record for {patient} on {date} flagged as INVALID (confidence: {confidence:.0%})"
        else:
            return f"✓ Record for {patient} on {date} validated as VALID (confidence: {confidence:.0%})"
    
    def _generate_layer2_reasoning(self, consensus: Dict, agent_results: Dict) -> List[str]:
        """Layer 2: Why agents made their decisions"""
        
        reasoning = []
        
        # Consensus type
        disagreement = consensus['disagreement_analysis']
        reasoning.append(f"**Consensus Type:** {disagreement['type']}")
        
        # Vote breakdown
        votes = consensus['weighted_votes']
        if consensus['final_decision'] == 'INVALID':
            reasoning.append(
                f"**Weighted Vote:** {votes['invalid_weight']:.2f} (INVALID) vs "
                f"{votes['valid_weight']:.2f} (VALID)"
            )
        else:
            reasoning.append(
                f"**Weighted Vote:** {votes['valid_weight']:.2f} (VALID) vs "
                f"{votes['invalid_weight']:.2f} (INVALID)"
            )
        
        # Individual agent reasoning
        reasoning.append("**Agent Decisions:**")
        
        for agent_name in ['trend', 'pattern', 'anomaly']:
            result = agent_results.get(agent_name, {})
            decision = "VALID" if result.get('valid', True) else "INVALID"
            confidence = result.get('confidence', 0)
            
            # Get agent's explanation
            explanation = result.get('explanation', 'No explanation provided')
            
            # Truncate long explanations
            if len(explanation) > 150:
                explanation = explanation[:150] + "..."
            
            reasoning.append(f"  - **{agent_name.capitalize()}**: {decision} ({confidence:.0%}) - {explanation}")
        
        return reasoning
    
    def _generate_layer3_evidence(self, consensus: Dict, agent_results: Dict, 
                                   record: Dict) -> Dict:
        """Layer 3: Detailed evidence and context"""
        
        evidence = {
            'record_details': self._format_record_details(record),
            'agent_evidence': {},
            'guardian_verification': {},
            'trust_weights': consensus.get('trust_weights', {}),
            'disagreement_details': consensus['disagreement_analysis']
        }
        
        # Gather detailed evidence from each agent
        for agent_name in ['trend', 'pattern', 'anomaly']:
            result = agent_results.get(agent_name, {})
            
            agent_evidence = {
                'decision': 'VALID' if result.get('valid', True) else 'INVALID',
                'confidence': result.get('confidence', 0)
            }
            
            # Trend-specific evidence
            if agent_name == 'trend' and 'issues' in result:
                agent_evidence['issues'] = result['issues']
            
            # Pattern-specific evidence
            if agent_name == 'pattern':
                if 'baseline' in result:
                    agent_evidence['patient_baseline'] = result['baseline']
                if 'deviations' in result:
                    agent_evidence['deviations'] = result['deviations']
            
            # Anomaly-specific evidence
            if agent_name == 'anomaly' and 'anomalies' in result:
                agent_evidence['anomalies'] = result['anomalies']
            
            evidence['agent_evidence'][agent_name] = agent_evidence
        
        # Guardian verification details
        guardian_result = agent_results.get('guardian', {})
        if guardian_result:
            evidence['guardian_verification'] = {
                'recommendation': guardian_result.get('recommendation', 'N/A'),
                'hard_limit_violations': guardian_result.get('hard_limit_violations', []),
                'needs_human_review': guardian_result.get('needs_human_review', False),
                'explanation': guardian_result.get('explanation', '')
            }
        
        return evidence
    
    def _format_record_details(self, record: Dict) -> Dict:
        """Format record data for display"""
        return {
            'date': record['record_date'],
            'patient_id': record['patient_id'],
            'vitals': {
                'weight': f"{record['weight_kg']}kg",
                'heart_rate': f"{record['heart_rate_bpm']}bpm",
                'blood_pressure': f"{record['systolic_bp_mmhg']}/{record['diastolic_bp_mmhg']}mmHg",
                'temperature': f"{record['temperature_celsius']}°C"
            }
        }
    
    def _generate_recommendation(self, consensus: Dict, agent_results: Dict) -> Dict:
        """Generate actionable recommendation"""
        
        guardian_rec = agent_results.get('guardian', {}).get('recommendation', 'N/A')
        
        if "REJECT" in guardian_rec:
            return {
                'action': 'REJECT',
                'reason': 'Critical data quality violation detected',
                'next_steps': [
                    'Do not use this record for clinical decisions',
                    'Verify measurement with patient or re-measure',
                    'Check equipment calibration',
                    'Review data entry procedures'
                ]
            }
        elif "HUMAN_REVIEW" in guardian_rec:
            return {
                'action': 'HUMAN_REVIEW',
                'reason': 'Uncertain or conflicting agent decisions',
                'next_steps': [
                    'Manual review recommended',
                    'Examine agent disagreement details',
                    'Verify against patient history',
                    'Consult clinical expert if needed'
                ]
            }
        elif consensus['final_decision'] == 'INVALID':
            return {
                'action': 'FLAG',
                'reason': 'Agent consensus indicates data quality issue',
                'next_steps': [
                    'Flag record for quality review',
                    'Investigate flagged fields',
                    'Compare with adjacent records in timeline'
                ]
            }
        else:
            return {
                'action': 'ACCEPT',
                'reason': 'Record passes all validation checks',
                'next_steps': [
                    'Record approved for use',
                    'No action required'
                ]
            }
    
    def _format_full_explanation(self, layer1: str, layer2: List[str], 
                                  layer3: Dict, recommendation: Dict) -> str:
        """Format complete explanation as readable text"""
        
        explanation = []
        
        # Layer 1
        explanation.append("=" * 80)
        explanation.append("LAYER 1: SUMMARY")
        explanation.append("=" * 80)
        explanation.append(layer1)
        explanation.append("")
        
        # Layer 2
        explanation.append("=" * 80)
        explanation.append("LAYER 2: REASONING")
        explanation.append("=" * 80)
        for line in layer2:
            explanation.append(line)
        explanation.append("")
        
        # Layer 3
        explanation.append("=" * 80)
        explanation.append("LAYER 3: EVIDENCE & CONTEXT")
        explanation.append("=" * 80)
        explanation.append(f"**Record Details:**")
        explanation.append(f"  Date: {layer3['record_details']['date']}")
        explanation.append(f"  Patient: {layer3['record_details']['patient_id']}")
        explanation.append(f"  Vitals: {layer3['record_details']['vitals']}")
        explanation.append("")
        
        # Guardian
        guardian = layer3['guardian_verification']
        explanation.append(f"**Guardian Recommendation:** {guardian.get('recommendation', 'N/A')}")
        if guardian.get('hard_limit_violations'):
            explanation.append(f"  ⚠️ Hard Limit Violations: {len(guardian['hard_limit_violations'])}")
        explanation.append("")
        
        # Recommendation
        explanation.append("=" * 80)
        explanation.append("RECOMMENDATION")
        explanation.append("=" * 80)
        explanation.append(f"**Action:** {recommendation['action']}")
        explanation.append(f"**Reason:** {recommendation['reason']}")
        explanation.append("**Next Steps:**")
        for step in recommendation['next_steps']:
            explanation.append(f"  - {step}")
        
        return "\n".join(explanation)
    
    def export_json(self, explanation: Dict, filename: str = None):
        """Export explanation to JSON file"""
        if filename:
            with open(filename, 'w') as f:
                json.dump(explanation, f, indent=2, default=str)
        return json.dumps(explanation, indent=2, default=str)

# Test Explanation Engine
if __name__ == "__main__":
    import sys
    import os
    import pandas as pd
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from agents.trend_validator import TrendValidator
    from agents.pattern_analyzer import PatternAnalyzer
    from agents.anomaly_detector import AnomalyDetector
    from agents.guardian_agent import GuardianAgent
    from core.consensus import TrustWeightedConsensus
    
    # Load data
    df_all = pd.read_csv('data/synthetic_ehr_test.csv')
    patient_1 = df_all[df_all['patient_id'] == 'P0001'].sort_values('record_date').reset_index(drop=True)
    
    # Initialize
    trend_agent = TrendValidator()
    pattern_agent = PatternAnalyzer()
    anomaly_agent = AnomalyDetector()
    anomaly_agent.fit_population_statistics(df_all)
    guardian_agent = GuardianAgent()
    consensus_engine = TrustWeightedConsensus()
    explanation_engine = ExplanationEngine()
    
    print(f"Testing {explanation_engine.name}...")
    print("="*80)
    
    # Process one problematic record (the one with HR 840)
    record = patient_1.iloc[3]  # 2024-02-15 with HR 840
    
    # Run all agents
    trend_results = trend_agent.validate_patient_timeline(patient_1)
    pattern_results = pattern_agent.validate_patient_timeline(patient_1)
    anomaly_results = anomaly_agent.validate_patient_timeline(patient_1)
    
    # Get results for this specific record
    record_date = record['record_date']
    trend_result = next((r for r in trend_results if r['record_date'] == record_date), {})
    pattern_result = next((r for r in pattern_results if r['record_date'] == record_date), {})
    anomaly_result = next((r for r in anomaly_results if r['record_date'] == record_date), {})
    
    agent_results = {
        'trend': trend_result,
        'pattern': pattern_result,
        'anomaly': anomaly_result
    }
    
    # Guardian verification
    guardian_result = guardian_agent.verify_agents_consensus(record, agent_results)
    agent_results['guardian'] = guardian_result
    
    # Consensus
    consensus = consensus_engine.compute_consensus(agent_results, record)
    
    # Generate explanation
    explanation = explanation_engine.generate_explanation(consensus, agent_results, record)
    
    # Display
    print(explanation['full_explanation'])
    
    print("\n" + "="*80)
    print("JSON Export Sample:")
    print("="*80)
    print(explanation_engine.export_json(explanation)[:500] + "...")