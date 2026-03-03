import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from agents.trend_validator import TrendValidator
from agents.pattern_analyzer import PatternAnalyzer
from agents.anomaly_detector import AnomalyDetector
from agents.guardian_agent import GuardianAgent
from core.consensus import TrustWeightedConsensus

class AdaptiveTrustEvaluator:
    """
    Evaluates system with adaptive trust learning
    Shows how trust scores evolve over time
    """
    
    def __init__(self):
        self.name = "Adaptive Trust Evaluator"
        
        # Initialize agents
        self.trend_agent = TrendValidator()
        self.pattern_agent = PatternAnalyzer()
        self.anomaly_agent = AnomalyDetector()
        self.guardian_agent = GuardianAgent()
        self.consensus_engine = TrustWeightedConsensus()
        
        # Track evolution
        self.trust_evolution = []
        self.accuracy_evolution = []
        
    def run_adaptive_learning(self, ground_truth_file='data/synthetic_ehr_full.csv',
                             learning_window=100):
        """
        Run evaluation with adaptive trust learning
        
        Parameters:
        - learning_window: Update trust every N records
        """
        print(f"{'='*80}")
        print(f"ADAPTIVE TRUST LEARNING EVALUATION")
        print(f"{'='*80}\n")
        
        # Load data
        df_full = pd.read_csv(ground_truth_file)
        print(f"✓ Loaded {len(df_full)} records")
        
        # Fit population stats
        self.anomaly_agent.fit_population_statistics(df_full)
        
        # Process in batches to show learning
        patients = df_full['patient_id'].unique()
        
        batch_results = []
        record_count = 0
        
        for patient_id in patients:
            patient_data = df_full[df_full['patient_id'] == patient_id].sort_values('record_date').reset_index(drop=True)
            
            # Run agents
            trend_results = self.trend_agent.validate_patient_timeline(patient_data)
            pattern_results = self.pattern_agent.validate_patient_timeline(patient_data)
            anomaly_results = self.anomaly_agent.validate_patient_timeline(patient_data)
            
            # Process each record
            for record_idx, record in patient_data.iterrows():
                record_date = record['record_date']
                ground_truth = record['has_error']
                error_type = record['error_type']
                
                # Get agent results
                trend_result = next((r for r in trend_results if r['record_date'] == record_date), {})
                pattern_result = next((r for r in pattern_results if r['record_date'] == record_date), {})
                anomaly_result = next((r for r in anomaly_results if r['record_date'] == record_date), {})
                
                agent_results = {
                    'trend': trend_result,
                    'pattern': pattern_result,
                    'anomaly': anomaly_result
                }
                
                # Guardian + Consensus (BEFORE feedback)
                guardian_result = self.guardian_agent.verify_agents_consensus(record, agent_results)
                agent_results['guardian'] = guardian_result
                consensus = self.consensus_engine.compute_consensus(agent_results, record)
                
                # ===== ADAPTIVE LEARNING: Give feedback =====
                consensus_correct = (consensus['final_decision'] == 'INVALID') == ground_truth
                
                # Update each agent's trust based on their individual correctness
                trend_correct = (not trend_result.get('valid', True)) == ground_truth
                pattern_correct = (not pattern_result.get('valid', True)) == ground_truth
                anomaly_correct = (not anomaly_result.get('valid', True)) == ground_truth
                
                self.consensus_engine.update_trust_with_feedback('trend', trend_correct, error_type)
                self.consensus_engine.update_trust_with_feedback('pattern', pattern_correct, error_type)
                self.consensus_engine.update_trust_with_feedback('anomaly', anomaly_correct, error_type)
                
                # Track results
                batch_results.append({
                    'record_num': record_count,
                    'ground_truth': ground_truth,
                    'consensus_correct': consensus_correct,
                    'trend_trust': self.consensus_engine.agent_trust_scores['trend'],
                    'pattern_trust': self.consensus_engine.agent_trust_scores['pattern'],
                    'anomaly_trust': self.consensus_engine.agent_trust_scores['anomaly']
                })
                
                record_count += 1
                
                # Snapshot every learning_window records
                if record_count % learning_window == 0:
                    self._take_snapshot(record_count, batch_results)
                    print(f"  Progress: {record_count}/1000 records | Trust: Trend={self.consensus_engine.agent_trust_scores['trend']:.3f}, Pattern={self.consensus_engine.agent_trust_scores['pattern']:.3f}, Anomaly={self.consensus_engine.agent_trust_scores['anomaly']:.3f}")
        
        # Final snapshot
        self._take_snapshot(record_count, batch_results)
        
        self.batch_results_df = pd.DataFrame(batch_results)
        
        print(f"\n✓ Adaptive learning complete!")
        print(f"\n{'='*80}")
        print("FINAL TRUST SCORES")
        print(f"{'='*80}")
        evolution = self.consensus_engine.get_trust_evolution_data()
        for agent, data in evolution.items():
            print(f"{agent.capitalize()}:")
            print(f"  Trust Score: {data['current_trust']:.3f}")
            print(f"  Accuracy: {data['accuracy']*100:.1f}%")
            print(f"  Decisions: {data['total_decisions']}")
        
        return self.batch_results_df
    
    def _take_snapshot(self, record_num, results):
        """Take snapshot of current state"""
        recent = results[-100:] if len(results) >= 100 else results
        accuracy = sum(r['consensus_correct'] for r in recent) / len(recent) if recent else 0
        
        self.trust_evolution.append({
            'record_num': record_num,
            'trend_trust': self.consensus_engine.agent_trust_scores['trend'],
            'pattern_trust': self.consensus_engine.agent_trust_scores['pattern'],
            'anomaly_trust': self.consensus_engine.agent_trust_scores['anomaly']
        })
        
        self.accuracy_evolution.append({
            'record_num': record_num,
            'accuracy': accuracy
        })
    
    def plot_trust_evolution(self, save_path='results/trust_evolution.png'):
        """Plot how trust scores evolved over time"""
        import os
        os.makedirs('results', exist_ok=True)
        
        df_trust = pd.DataFrame(self.trust_evolution)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(df_trust['record_num'], df_trust['trend_trust'], 
                label='Trend Validator', marker='o', linewidth=2)
        ax.plot(df_trust['record_num'], df_trust['pattern_trust'], 
                label='Pattern Analyzer', marker='s', linewidth=2)
        ax.plot(df_trust['record_num'], df_trust['anomaly_trust'], 
                label='Anomaly Detector', marker='^', linewidth=2)
        
        ax.set_xlabel('Records Processed', fontweight='bold', fontsize=12)
        ax.set_ylabel('Trust Score', fontweight='bold', fontsize=12)
        ax.set_title('Adaptive Trust Score Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Trust evolution saved to {save_path}")
        plt.close()
    
    def plot_accuracy_evolution(self, save_path='results/accuracy_evolution.png'):
        """Plot how accuracy improved with adaptive trust"""
        import os
        os.makedirs('results', exist_ok=True)
        
        df_acc = pd.DataFrame(self.accuracy_evolution)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(df_acc['record_num'], df_acc['accuracy'], 
                color='#e74c3c', linewidth=2, marker='o')
        
        ax.set_xlabel('Records Processed', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy (Rolling 100 records)', fontweight='bold', fontsize=12)
        ax.set_title('System Accuracy with Adaptive Trust Learning', 
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        # Add trend line
        z = np.polyfit(df_acc['record_num'], df_acc['accuracy'], 1)
        p = np.poly1d(z)
        ax.plot(df_acc['record_num'], p(df_acc['record_num']), 
                "--", color='gray', alpha=0.8, label=f'Trend (slope={z[0]:.4f})')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Accuracy evolution saved to {save_path}")
        plt.close()


# Run adaptive evaluation
if __name__ == "__main__":
    evaluator = AdaptiveTrustEvaluator()
    
    # Run with adaptive learning
    results_df = evaluator.run_adaptive_learning(learning_window=100)
    
    # Plot evolution
    evaluator.plot_trust_evolution()
    evaluator.plot_accuracy_evolution()
    
    # Save results
    results_df.to_csv('results/adaptive_learning_results.csv', index=False)
    print(f"✓ Results saved to results/adaptive_learning_results.csv")
    
    print(f"\n{'='*80}")
    print("ADAPTIVE TRUST EVALUATION COMPLETE")
    print(f"{'='*80}")