import pandas as pd
import numpy as np
from typing import Dict, List

class TrustWeightedConsensus:
    """
    Combines decisions from multiple agents using trust-weighted voting
    Trust weights are based on historical accuracy per agent
    """
    
    def __init__(self):
        self.name = "Trust-Weighted Consensus Engine"
        
        # Initialize trust scores (start equal, will adapt based on performance)
        self.agent_trust_scores = {
            'trend': 1.0,
            'pattern': 1.0,
            'anomaly': 1.0
        }
        
        # Track agent performance history
        self.agent_history = {
            'trend': {'correct': 0, 'total': 0, 'by_error_type': {}},
            'pattern': {'correct': 0, 'total': 0, 'by_error_type': {}},
            'anomaly': {'correct': 0, 'total': 0, 'by_error_type': {}}
        }
    
    def compute_consensus(self, agent_results: Dict, record: pd.Series) -> Dict:
        """
        Compute weighted consensus from all agent results
        
        Parameters:
        - agent_results: Dict with keys 'trend', 'pattern', 'anomaly', 'guardian'
        - record: The actual data record being validated
        
        Returns:
        - Consensus decision with explanation
        """
        
        # Extract agent decisions and confidences
        agent_decisions = {}
        agent_confidences = {}
        
        for agent_name in ['trend', 'pattern', 'anomaly']:
            result = agent_results.get(agent_name, {})
            agent_decisions[agent_name] = result.get('valid', True)
            agent_confidences[agent_name] = result.get('confidence', 0.5)
        
        # Calculate weighted votes
        weighted_votes = self._calculate_weighted_votes(
            agent_decisions, 
            agent_confidences
        )
        
        # Determine final decision
        final_decision = weighted_votes['invalid_weight'] > weighted_votes['valid_weight']
        
        # Overall confidence
        if final_decision:  # Invalid
            confidence = weighted_votes['invalid_weight'] / weighted_votes['total_weight']
        else:  # Valid
            confidence = weighted_votes['valid_weight'] / weighted_votes['total_weight']
        
        # Get Guardian's verification
        guardian_result = agent_results.get('guardian', {})
        
        # Analyze disagreements
        disagreement_analysis = self._analyze_disagreements(
            agent_decisions, 
            agent_confidences,
            agent_results
        )
        
        # Generate consensus explanation
        explanation = self._generate_consensus_explanation(
            agent_decisions,
            weighted_votes,
            final_decision,
            disagreement_analysis,
            guardian_result
        )
        
        return {
            'record_date': record['record_date'],
            'patient_id': record['patient_id'],
            'final_decision': 'INVALID' if final_decision else 'VALID',
            'confidence': round(confidence, 2),
            'agent_decisions': agent_decisions,
            'agent_confidences': agent_confidences,
            'trust_weights': self.agent_trust_scores.copy(),
            'weighted_votes': weighted_votes,
            'disagreement_analysis': disagreement_analysis,
            'guardian_verification': guardian_result.get('recommendation', 'N/A'),
            'explanation': explanation
        }
    
    def _calculate_weighted_votes(self, decisions: Dict, confidences: Dict) -> Dict:
        """Calculate weighted votes based on trust scores and confidence"""
        
        valid_weight = 0
        invalid_weight = 0
        
        for agent, decision in decisions.items():
            # Weight = Trust Score × Confidence
            trust = self.agent_trust_scores[agent]
            confidence = confidences[agent]
            weight = trust * confidence
            
            if decision:  # Valid
                valid_weight += weight
            else:  # Invalid
                invalid_weight += weight
        
        total_weight = valid_weight + invalid_weight
        
        return {
            'valid_weight': round(valid_weight, 3),
            'invalid_weight': round(invalid_weight, 3),
            'total_weight': round(total_weight, 3),
            'vote_breakdown': {
                agent: {
                    'decision': 'VALID' if decisions[agent] else 'INVALID',
                    'confidence': confidences[agent],
                    'trust': self.agent_trust_scores[agent],
                    'weight': round(self.agent_trust_scores[agent] * confidences[agent], 3)
                }
                for agent in decisions.keys()
            }
        }
    
    def _analyze_disagreements(self, decisions: Dict, confidences: Dict, 
                               agent_results: Dict) -> Dict:
        """Analyze when and why agents disagree"""
        
        # Count votes
        invalid_count = sum(1 for d in decisions.values() if not d)
        valid_count = sum(1 for d in decisions.values() if d)
        
        disagreement_type = None
        disagreement_details = []
        
        if invalid_count == 3:
            disagreement_type = "UNANIMOUS_INVALID"
        elif invalid_count == 0:
            disagreement_type = "UNANIMOUS_VALID"
        elif invalid_count == 2:
            disagreement_type = "MAJORITY_INVALID"
            # Find the dissenting agent
            dissenting_agent = [a for a, d in decisions.items() if d][0]
            disagreement_details.append({
                'dissenting_agent': dissenting_agent,
                'reason': f"{dissenting_agent} marked VALID while others flagged issues"
            })
        elif invalid_count == 1:
            disagreement_type = "MAJORITY_VALID"
            # Find the flagging agent
            flagging_agent = [a for a, d in decisions.items() if not d][0]
            disagreement_details.append({
                'flagging_agent': flagging_agent,
                'reason': f"{flagging_agent} flagged while others marked VALID"
            })
        
        return {
            'type': disagreement_type,
            'invalid_count': invalid_count,
            'valid_count': valid_count,
            'has_disagreement': invalid_count not in [0, 3],
            'details': disagreement_details
        }
    
    def _generate_consensus_explanation(self, decisions, weighted_votes, 
                                       final_decision, disagreement, guardian):
        """Generate human-readable consensus explanation"""
        
        explanation_parts = []
        
        # Part 1: Consensus type
        explanation_parts.append(f"Consensus: {disagreement['type']}")
        
        # Part 2: Weighted decision
        if final_decision:
            explanation_parts.append(
                f"Weighted vote: {weighted_votes['invalid_weight']:.2f} (INVALID) vs "
                f"{weighted_votes['valid_weight']:.2f} (VALID) → Record FLAGGED"
            )
        else:
            explanation_parts.append(
                f"Weighted vote: {weighted_votes['valid_weight']:.2f} (VALID) vs "
                f"{weighted_votes['invalid_weight']:.2f} (INVALID) → Record ACCEPTED"
            )
        
        # Part 3: Agent breakdown
        breakdown = "Agent votes: "
        votes = []
        for agent, info in weighted_votes['vote_breakdown'].items():
            votes.append(f"{agent.capitalize()}({info['decision']}, weight={info['weight']})")
        breakdown += ", ".join(votes)
        explanation_parts.append(breakdown)
        
        # Part 4: Disagreement details
        if disagreement['has_disagreement']:
            explanation_parts.append(f"⚠️ Disagreement detected: {disagreement['details']}")
        
        # Part 5: Guardian recommendation
        if guardian.get('recommendation'):
            explanation_parts.append(f"Guardian: {guardian['recommendation']}")
        
        return " | ".join(explanation_parts)
    
    # ========== ADAPTIVE TRUST METHODS (NEW) ==========
    
    def update_trust_with_feedback(self, agent_name: str, was_correct: bool, 
                                    error_type: str = None):
        """
        Update agent trust score based on feedback with error-type awareness
        
        Parameters:
        - agent_name: 'trend', 'pattern', or 'anomaly'
        - was_correct: Boolean - was the agent's decision correct?
        - error_type: Optional - type of error (for specialized tracking)
        """
        if agent_name not in self.agent_history:
            return
        
        # Update overall history
        self.agent_history[agent_name]['total'] += 1
        if was_correct:
            self.agent_history[agent_name]['correct'] += 1
        
        # Update error-type-specific history
        if error_type:
            if error_type not in self.agent_history[agent_name]['by_error_type']:
                self.agent_history[agent_name]['by_error_type'][error_type] = {
                    'correct': 0, 'total': 0
                }
            
            self.agent_history[agent_name]['by_error_type'][error_type]['total'] += 1
            if was_correct:
                self.agent_history[agent_name]['by_error_type'][error_type]['correct'] += 1
        
        # Recalculate trust score with smooth learning
        history = self.agent_history[agent_name]
        if history['total'] > 0:
            accuracy = history['correct'] / history['total']
            
            # Trust formula: base (0.5) + accuracy bonus (0.5) with learning rate
            learning_rate = 0.1  # How quickly trust adapts
            current_trust = self.agent_trust_scores[agent_name]
            target_trust = 0.5 + (accuracy * 0.5)
            
            # Smooth update (not instant jump)
            self.agent_trust_scores[agent_name] = (
                current_trust * (1 - learning_rate) + target_trust * learning_rate
            )
            
            # Clamp between 0.3 and 1.0
            self.agent_trust_scores[agent_name] = max(0.3, min(1.0, 
                self.agent_trust_scores[agent_name]))
    
    def get_trust_evolution_data(self):
        """Get historical trust scores for visualization"""
        return {
            agent: {
                'current_trust': self.agent_trust_scores[agent],
                'accuracy': self.agent_history[agent]['correct'] / 
                           max(1, self.agent_history[agent]['total']),
                'total_decisions': self.agent_history[agent]['total'],
                'by_error_type': self.agent_history[agent].get('by_error_type', {})
            }
            for agent in self.agent_trust_scores.keys()
        }
    
    def reset_trust_scores(self):
        """Reset trust scores to initial state (for experiments)"""
        self.agent_trust_scores = {
            'trend': 1.0,
            'pattern': 1.0,
            'anomaly': 1.0
        }
        self.agent_history = {
            'trend': {'correct': 0, 'total': 0, 'by_error_type': {}},
            'pattern': {'correct': 0, 'total': 0, 'by_error_type': {}},
            'anomaly': {'correct': 0, 'total': 0, 'by_error_type': {}}
        }
    
    def get_trust_report(self) -> str:
        """Generate trust score report"""
        report = "Agent Trust Scores:\n"
        for agent, score in self.agent_trust_scores.items():
            history = self.agent_history[agent]
            accuracy = (history['correct'] / history['total'] * 100) if history['total'] > 0 else 0
            report += f"  {agent.capitalize()}: {score:.3f} (accuracy: {accuracy:.1f}%, n={history['total']})\n"
        return report


# Test Consensus Engine
if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from agents.trend_validator import TrendValidator
    from agents.pattern_analyzer import PatternAnalyzer
    from agents.anomaly_detector import AnomalyDetector
    from agents.guardian_agent import GuardianAgent
    
    # Load data
    df_all = pd.read_csv('data/synthetic_ehr_test.csv')
    patient_1 = df_all[df_all['patient_id'] == 'P0001'].sort_values('record_date').reset_index(drop=True)
    
    # Initialize all agents
    trend_agent = TrendValidator()
    pattern_agent = PatternAnalyzer()
    anomaly_agent = AnomalyDetector()
    anomaly_agent.fit_population_statistics(df_all)
    guardian_agent = GuardianAgent()
    consensus_engine = TrustWeightedConsensus()
    
    print(f"Testing {consensus_engine.name}...")
    print("="*80)
    
    # Run all agents
    trend_results = trend_agent.validate_patient_timeline(patient_1)
    pattern_results = pattern_agent.validate_patient_timeline(patient_1)
    anomaly_results = anomaly_agent.validate_patient_timeline(patient_1)
    
    # Process each record through consensus
    for idx, record in patient_1.iterrows():
        record_date = record['record_date']
        
        # Gather agent results
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
        
        # Consensus decision
        consensus = consensus_engine.compute_consensus(agent_results, record)
        
        print(f"\n{record_date}: {consensus['final_decision']} (confidence: {consensus['confidence']})")
        print(f"  {consensus['explanation']}")
        
        if consensus['disagreement_analysis']['has_disagreement']:
            print(f"   Disagreement: {consensus['disagreement_analysis']['type']}")
    
    print("\n" + "="*80)
    print(consensus_engine.get_trust_report())