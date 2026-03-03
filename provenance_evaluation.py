import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from agents.trend_validator import TrendValidator
from agents.pattern_analyzer import PatternAnalyzer
from agents.anomaly_detector import AnomalyDetector
from agents.guardian_agent import GuardianAgent
from core.consensus import TrustWeightedConsensus
from provenance_aware_validator import ProvenanceAwareValidator


class ProvenanceEvaluator:
    """
    Evaluate system with and without provenance awareness
    Compare performance to show provenance impact
    """
    
    def __init__(self):
        self.name = "Provenance Impact Evaluator"
    
    def evaluate_with_provenance(self, ground_truth_file='data/synthetic_ehr_full_provenance.csv'):
        """
        Run evaluation comparing standard vs provenance-aware validation
        """
        print(f"{'='*80}")
        print(f"PROVENANCE IMPACT EVALUATION")
        print(f"{'='*80}\n")
        
        # Load data with provenance
        df_full = pd.read_csv(ground_truth_file)
        print(f"✓ Loaded {len(df_full)} records with provenance metadata")
        
        # Initialize standard agents
        trend_agent = TrendValidator()
        pattern_agent = PatternAnalyzer()
        anomaly_agent = AnomalyDetector()
        guardian_agent = GuardianAgent()
        
        # Initialize provenance-aware agents
        provenance_trend = ProvenanceAwareValidator(TrendValidator())
        provenance_pattern = ProvenanceAwareValidator(PatternAnalyzer())
        provenance_anomaly = ProvenanceAwareValidator(AnomalyDetector())
        
        # Fit population statistics
        anomaly_agent.fit_population_statistics(df_full)
        provenance_anomaly.base_agent.fit_population_statistics(df_full)
        
        # Storage for results
        standard_results = []
        provenance_results = []
        
        patients = df_full['patient_id'].unique()[:20]
        print(f"✓ Processing {len(patients)} patients...\n")
        
        for idx, patient_id in enumerate(patients, 1):
            if idx % 20 == 0:
                print(f"  Progress: {idx}/{len(patients)} patients")
            
            patient_data = df_full[df_full['patient_id'] == patient_id].sort_values('record_date').reset_index(drop=True)
            
            # ===== STANDARD AGENTS =====
            std_trend = trend_agent.validate_patient_timeline(patient_data)
            std_pattern = pattern_agent.validate_patient_timeline(patient_data)
            std_anomaly = anomaly_agent.validate_patient_timeline(patient_data)
            
            # ===== PROVENANCE-AWARE AGENTS =====
            prov_trend = provenance_trend.validate_patient_timeline(patient_data)
            prov_pattern = provenance_pattern.validate_patient_timeline(patient_data)
            prov_anomaly = provenance_anomaly.validate_patient_timeline(patient_data)
            
            # Process each record
            for record_idx, record in patient_data.iterrows():
                record_date = record['record_date']
                ground_truth = record['has_error']
                source = record['source']
                
                # Get results for this record
                std_t = next((r for r in std_trend if r['record_date'] == record_date), {})
                std_p = next((r for r in std_pattern if r['record_date'] == record_date), {})
                std_a = next((r for r in std_anomaly if r['record_date'] == record_date), {})
                
                prov_t = next((r for r in prov_trend if r['record_date'] == record_date), {})
                prov_p = next((r for r in prov_pattern if r['record_date'] == record_date), {})
                prov_a = next((r for r in prov_anomaly if r['record_date'] == record_date), {})
                
                # Standard consensus
                std_consensus_engine = TrustWeightedConsensus()
                std_agent_results = {'trend': std_t, 'pattern': std_p, 'anomaly': std_a}
                std_guardian = guardian_agent.verify_agents_consensus(record, std_agent_results)
                std_agent_results['guardian'] = std_guardian
                std_consensus = std_consensus_engine.compute_consensus(std_agent_results, record)
                
                # Provenance-aware consensus
                prov_consensus_engine = TrustWeightedConsensus()
                prov_agent_results = {'trend': prov_t, 'pattern': prov_p, 'anomaly': prov_a}
                prov_guardian = guardian_agent.verify_agents_consensus(record, prov_agent_results)
                prov_agent_results['guardian'] = prov_guardian
                prov_consensus = prov_consensus_engine.compute_consensus(prov_agent_results, record)
                
                # Store results
                standard_results.append({
                    'ground_truth': ground_truth,
                    'consensus_flagged': std_consensus['final_decision'] == 'INVALID',
                    'consensus_confidence': std_consensus['confidence'],
                    'source': source
                })
                
                provenance_results.append({
                    'ground_truth': ground_truth,
                    'consensus_flagged': prov_consensus['final_decision'] == 'INVALID',
                    'consensus_confidence': prov_consensus['confidence'],
                    'source': source
                })
        
        print(f"✓ Evaluation complete!\n")
        
        # Convert to DataFrames
        self.std_df = pd.DataFrame(standard_results)
        self.prov_df = pd.DataFrame(provenance_results)
        
        return self.std_df, self.prov_df
    
    def compare_performance(self):
        """Compare standard vs provenance-aware performance"""
        
        print(f"{'='*80}")
        print(f"PERFORMANCE COMPARISON")
        print(f"{'='*80}\n")
        
        # Overall metrics
        y_true = self.std_df['ground_truth'].astype(int)
        
        # Standard
        std_pred = self.std_df['consensus_flagged'].astype(int)
        std_precision = precision_score(y_true, std_pred, zero_division=0)
        std_recall = recall_score(y_true, std_pred, zero_division=0)
        std_f1 = f1_score(y_true, std_pred, zero_division=0)
        
        # Provenance-aware
        prov_pred = self.prov_df['consensus_flagged'].astype(int)
        prov_precision = precision_score(y_true, prov_pred, zero_division=0)
        prov_recall = recall_score(y_true, prov_pred, zero_division=0)
        prov_f1 = f1_score(y_true, prov_pred, zero_division=0)
        
        print("STANDARD CONSENSUS:")
        print(f"  Precision: {std_precision:.3f}")
        print(f"  Recall:    {std_recall:.3f}")
        print(f"  F1-Score:  {std_f1:.3f}\n")
        
        print("PROVENANCE-AWARE CONSENSUS:")
        print(f"  Precision: {prov_precision:.3f}")
        print(f"  Recall:    {prov_recall:.3f}")
        print(f"  F1-Score:  {prov_f1:.3f}\n")
        
        print(f"{'='*80}")
        print("IMPROVEMENT:")
        print(f"{'='*80}")
        print(f"  Precision: {(prov_precision - std_precision)*100:+.1f}%")
        print(f"  Recall:    {(prov_recall - std_recall)*100:+.1f}%")
        print(f"  F1-Score:  {(prov_f1 - std_f1)*100:+.1f}%\n")
        
        # Performance by source
        print(f"{'='*80}")
        print("PERFORMANCE BY DATA SOURCE")
        print(f"{'='*80}\n")
        
        for source in ['manual_entry', 'sensor', 'lab_verified']:
            std_source = self.std_df[self.std_df['source'] == source]
            prov_source = self.prov_df[self.prov_df['source'] == source]
            
            if len(std_source) == 0:
                continue
            
            y_true_source = std_source['ground_truth'].astype(int)
            
            std_pred_source = std_source['consensus_flagged'].astype(int)
            prov_pred_source = prov_source['consensus_flagged'].astype(int)
            
            std_f1_source = f1_score(y_true_source, std_pred_source, zero_division=0)
            prov_f1_source = f1_score(y_true_source, prov_pred_source, zero_division=0)
            
            print(f"{source.upper().replace('_', ' ')}:")
            print(f"  Standard F1:    {std_f1_source:.3f}")
            print(f"  Provenance F1:  {prov_f1_source:.3f}")
            print(f"  Improvement:    {(prov_f1_source - std_f1_source)*100:+.1f}%\n")
    
    def plot_provenance_impact(self, save_path='results/provenance_impact.png'):
        """Visualize provenance impact"""
        import os
        os.makedirs('results', exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Overall comparison
        y_true = self.std_df['ground_truth'].astype(int)
        std_pred = self.std_df['consensus_flagged'].astype(int)
        prov_pred = self.prov_df['consensus_flagged'].astype(int)
        
        std_precision = precision_score(y_true, std_pred, zero_division=0)
        std_recall = recall_score(y_true, std_pred, zero_division=0)
        std_f1 = f1_score(y_true, std_pred, zero_division=0)
        
        prov_precision = precision_score(y_true, prov_pred, zero_division=0)
        prov_recall = recall_score(y_true, prov_pred, zero_division=0)
        prov_f1 = f1_score(y_true, prov_pred, zero_division=0)
        
        metrics = ['Precision', 'Recall', 'F1-Score']
        standard = [std_precision, std_recall, std_f1]
        provenance = [prov_precision, prov_recall, prov_f1]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0].bar(x - width/2, standard, width, label='Standard', color='#3498db')
        axes[0].bar(x + width/2, provenance, width, label='Provenance-Aware', color='#2ecc71')
        
        axes[0].set_ylabel('Score')
        axes[0].set_title('Overall Performance Comparison', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1.1])
        
        # Plot 2: F1-Score by source
        sources = ['manual_entry', 'sensor', 'lab_verified']
        source_labels = ['Manual Entry', 'Sensor', 'Lab Verified']
        std_f1_by_source = []
        prov_f1_by_source = []
        
        for source in sources:
            std_source = self.std_df[self.std_df['source'] == source]
            prov_source = self.prov_df[self.prov_df['source'] == source]
            
            y_true_source = std_source['ground_truth'].astype(int)
            std_pred_source = std_source['consensus_flagged'].astype(int)
            prov_pred_source = prov_source['consensus_flagged'].astype(int)
            
            std_f1_by_source.append(f1_score(y_true_source, std_pred_source, zero_division=0))
            prov_f1_by_source.append(f1_score(y_true_source, prov_pred_source, zero_division=0))
        
        x = np.arange(len(sources))
        axes[1].bar(x - width/2, std_f1_by_source, width, label='Standard', color='#3498db')
        axes[1].bar(x + width/2, prov_f1_by_source, width, label='Provenance-Aware', color='#2ecc71')
        
        axes[1].set_ylabel('F1-Score')
        axes[1].set_title('F1-Score by Data Source', fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(source_labels, rotation=15, ha='right')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Provenance impact visualization saved to {save_path}")
        plt.close()


# Run evaluation
if __name__ == "__main__":
    evaluator = ProvenanceEvaluator()
    
    # Run evaluation
    std_df, prov_df = evaluator.evaluate_with_provenance()
    
    # Compare performance
    evaluator.compare_performance()
    
    # Visualize
    evaluator.plot_provenance_impact()
    
    # Save results
    std_df.to_csv('results/standard_results.csv', index=False)
    prov_df.to_csv('results/provenance_results.csv', index=False)
    print(f"✓ Results saved to results/")
    
    print(f"\n{'='*80}")
    print("PROVENANCE EVALUATION COMPLETE")
    print(f"{'='*80}")