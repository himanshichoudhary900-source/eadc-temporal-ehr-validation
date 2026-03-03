import google.generativeai as genai
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GuardianAgent:
    """
    Agent 4: Meta-validator (Guardian)
    Verifies decisions from other agents
    Ensures evidence-backed conclusions
    Detects hallucinations or unsupported inferences
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.name = "Guardian Agent"
        
        # Physiological hard limits (absolute boundaries)
        self.absolute_limits = {
            'weight_kg': (20, 300),
            'heart_rate_bpm': (30, 220),
            'systolic_bp_mmhg': (50, 250),
            'diastolic_bp_mmhg': (30, 150),
            'temperature_celsius': (32, 43)
        }
        
        # Confidence threshold for human review
        self.human_review_threshold = 0.7
    
    def verify_agents_consensus(self, record, agent_results):
        """
        Verify decisions from all 3 agents
        
        Parameters:
        - record: The actual data record being validated
        - agent_results: Dict with results from Agent 1, 2, 3
          {
              'trend': {...},
              'pattern': {...},
              'anomaly': {...}
          }
        
        Returns:
        - Verification result
        """
        
        # Extract agent decisions
        agent_decisions = {
            'trend': agent_results.get('trend', {}).get('valid', True),
            'pattern': agent_results.get('pattern', {}).get('valid', True),
            'anomaly': agent_results.get('anomaly', {}).get('valid', True)
        }
        
        agent_confidences = {
            'trend': agent_results.get('trend', {}).get('confidence', 0.5),
            'pattern': agent_results.get('pattern', {}).get('confidence', 0.5),
            'anomaly': agent_results.get('anomaly', {}).get('confidence', 0.5)
        }
        
        # Check 1: Verify against absolute physiological limits
        hard_limit_violations = self._check_hard_limits(record)
        
        # Check 2: Verify agent reasoning is evidence-backed
        reasoning_check = self._verify_agent_reasoning(record, agent_results)
        
        # Check 3: Detect consensus or conflicts
        consensus_analysis = self._analyze_consensus(agent_decisions, agent_confidences)
        
        # Check 4: Trigger human review if needed
        needs_human_review = self._should_trigger_human_review(
            consensus_analysis,
            agent_confidences,
            hard_limit_violations
        )
        
        # Generate final verification
        verification = self._generate_verification(
            record,
            agent_decisions,
            consensus_analysis,
            hard_limit_violations,
            reasoning_check,
            needs_human_review
        )
        
        return verification
    
    def _check_hard_limits(self, record):
        """Check if values violate absolute physiological limits"""
        violations = []
        
        for field, (min_val, max_val) in self.absolute_limits.items():
            value = record[field]
            
            if value < min_val or value > max_val:
                violations.append({
                    'field': field,
                    'value': value,
                    'limit_min': min_val,
                    'limit_max': max_val,
                    'severity': 'CRITICAL'
                })
        
        return violations
    
    def _verify_agent_reasoning(self, record, agent_results):
        """Verify that agent reasoning is supported by evidence"""
        issues = []
        
        # Check each agent's flagged issues have concrete evidence
        for agent_name in ['trend', 'pattern', 'anomaly']:
            agent_result = agent_results.get(agent_name, {})
            
            if not agent_result.get('valid', True):
                # Agent flagged something - verify it has evidence
                has_evidence = False
                
                if agent_name == 'trend' and 'issues' in agent_result:
                    has_evidence = len(agent_result['issues']) > 0
                elif agent_name == 'pattern' and 'deviations' in agent_result:
                    has_evidence = len(agent_result['deviations']) > 0
                elif agent_name == 'anomaly' and 'anomalies' in agent_result:
                    has_evidence = len(agent_result['anomalies']) > 0
                
                if not has_evidence:
                    issues.append({
                        'agent': agent_name,
                        'problem': 'Flagged as invalid but no specific evidence provided',
                        'severity': 'WARNING'
                    })
        
        return {
            'all_evidence_backed': len(issues) == 0,
            'issues': issues
        }
    
    def _analyze_consensus(self, agent_decisions, agent_confidences):
        """Analyze agreement/disagreement among agents"""
        
        # Count votes
        invalid_votes = sum(1 for valid in agent_decisions.values() if not valid)
        valid_votes = sum(1 for valid in agent_decisions.values() if valid)
        
        # Calculate weighted consensus
        total_weight = 0
        invalid_weight = 0
        
        for agent, decision in agent_decisions.items():
            confidence = agent_confidences[agent]
            total_weight += confidence
            if not decision:  # Invalid
                invalid_weight += confidence
        
        weighted_invalid_ratio = invalid_weight / total_weight if total_weight > 0 else 0
        
        # Determine consensus type
        if invalid_votes == 3:
            consensus_type = "UNANIMOUS_INVALID"
            final_decision = False
            confidence = 0.95
        elif invalid_votes == 2:
            consensus_type = "MAJORITY_INVALID"
            final_decision = False
            confidence = 0.85
        elif invalid_votes == 1:
            consensus_type = "SPLIT_DECISION"
            final_decision = weighted_invalid_ratio > 0.5  # Use weighted
            confidence = 0.70
        else:  # invalid_votes == 0
            consensus_type = "UNANIMOUS_VALID"
            final_decision = True
            confidence = 0.90
        
        return {
            'consensus_type': consensus_type,
            'final_decision': final_decision,
            'confidence': round(confidence, 2),
            'invalid_votes': invalid_votes,
            'valid_votes': valid_votes,
            'weighted_invalid_ratio': round(weighted_invalid_ratio, 2),
            'agent_decisions': agent_decisions
        }
    
    def _should_trigger_human_review(self, consensus, agent_confidences, hard_limit_violations):
        """Determine if human review is needed"""
        
        # Trigger if:
        # 1. Split decision (1-2 agents disagree)
        # 2. Low confidence from any agent
        # 3. Hard limit violations
        
        if consensus['consensus_type'] == 'SPLIT_DECISION':
            return True
        
        if any(conf < self.human_review_threshold for conf in agent_confidences.values()):
            return True
        
        if len(hard_limit_violations) > 0:
            return True
        
        return False
    
    def _generate_verification(self, record, agent_decisions, consensus, 
                               hard_limit_violations, reasoning_check, needs_human_review):
        """Generate final verification result with LLM explanation"""
        
        # Build context for LLM
        context = f"""
        Record Date: {record['record_date']}
        Patient: {record['patient_id']}
        
        Values:
        - Weight: {record['weight_kg']}kg
        - Heart Rate: {record['heart_rate_bpm']}bpm
        - BP: {record['systolic_bp_mmhg']}/{record['diastolic_bp_mmhg']}mmHg
        - Temperature: {record['temperature_celsius']}°C
        
        Agent Decisions:
        - Trend Validator: {'FLAGGED' if not agent_decisions['trend'] else 'VALID'}
        - Pattern Analyzer: {'FLAGGED' if not agent_decisions['pattern'] else 'VALID'}
        - Anomaly Detector: {'FLAGGED' if not agent_decisions['anomaly'] else 'VALID'}
        
        Consensus: {consensus['consensus_type']}
        Hard Limit Violations: {len(hard_limit_violations)}
        """
        
        if hard_limit_violations:
            context += "\nViolations:\n"
            for v in hard_limit_violations:
                context += f"- {v['field']}: {v['value']} (limits: {v['limit_min']}-{v['limit_max']})\n"
        
        prompt = f"""
        You are the Guardian Agent verifying other agents' decisions.
        
        {context}
        
        Provide a brief verification summary (2-3 sentences):
        1. Do you agree with the agents' consensus?
        2. Are there any concerns or hallucinations detected?
        3. Final recommendation (approve, flag, or human review needed)
        
        Be concise and authoritative.
        """
        
        try:
            response = self.model.generate_content(prompt)
            explanation = response.text.strip()
        except Exception as e:
            explanation = f"Verification complete. Consensus: {consensus['consensus_type']}"
        
        return {
            'verified': True,
            'final_decision': 'INVALID' if not consensus['final_decision'] else 'VALID',
            'consensus': consensus,
            'hard_limit_violations': hard_limit_violations,
            'reasoning_verified': reasoning_check['all_evidence_backed'],
            'needs_human_review': needs_human_review,
            'confidence': consensus['confidence'],
            'explanation': explanation,
            'recommendation': self._get_recommendation(consensus, needs_human_review, hard_limit_violations)
        }
    
    def _get_recommendation(self, consensus, needs_human_review, violations):
        """Generate action recommendation"""
        if len(violations) > 0:
            return "REJECT - Critical physiological limit violation"
        elif needs_human_review:
            return "HUMAN_REVIEW - Uncertain or split decision"
        elif not consensus['final_decision']:
            return "FLAG - Agents consensus indicates error"
        else:
            return "ACCEPT - Agents consensus indicates valid"

# Test Guardian Agent
if __name__ == "__main__":
    from trend_validator import TrendValidator
    from pattern_analyzer import PatternAnalyzer
    from anomaly_detector import AnomalyDetector
    
    # Load data
    df_all = pd.read_csv('data/synthetic_ehr_test.csv')
    patient_1 = df_all[df_all['patient_id'] == 'P0001'].sort_values('record_date')
    
    # Initialize all agents
    trend_agent = TrendValidator()
    pattern_agent = PatternAnalyzer()
    anomaly_agent = AnomalyDetector()
    anomaly_agent.fit_population_statistics(df_all)
    guardian = GuardianAgent()
    
    print(f"Testing {guardian.name}...")
    print("="*80)
    
    # Validate timeline with all agents
    trend_results = trend_agent.validate_patient_timeline(patient_1)
    pattern_results = pattern_agent.validate_patient_timeline(patient_1)
    anomaly_results = anomaly_agent.validate_patient_timeline(patient_1)
    
    # Guardian verifies each record
    for idx, record in patient_1.iterrows():
        record_date = record['record_date']
        
        # Get results for this record
        trend_result = next((r for r in trend_results if r['record_date'] == record_date), {})
        pattern_result = next((r for r in pattern_results if r['record_date'] == record_date), {})
        anomaly_result = next((r for r in anomaly_results if r['record_date'] == record_date), {})
        
        agent_results = {
            'trend': trend_result,
            'pattern': pattern_result,
            'anomaly': anomaly_result
        }
        
        # Guardian verification
        verification = guardian.verify_agents_consensus(record, agent_results)
        
        print(f"\n{record_date}: {verification['final_decision']}")
        print(f"  Consensus: {verification['consensus']['consensus_type']} (confidence: {verification['confidence']})")
        print(f"  Recommendation: {verification['recommendation']}")
        print(f"  {verification['explanation']}")