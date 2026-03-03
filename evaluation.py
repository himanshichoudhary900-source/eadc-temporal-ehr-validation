import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from agents.trend_validator import TrendValidator
from agents.pattern_analyzer import PatternAnalyzer
from agents.anomaly_detector import AnomalyDetector
from agents.guardian_agent import GuardianAgent
from core.consensus import TrustWeightedConsensus

class EHRValidationEvaluator:
    """
    Evaluates the multi-agent validation system against ground truth
    """
    
    def __init__(self):
        self.name = "EHR Validation Evaluator"
        
        # Initialize agents
        self.trend_agent = TrendValidator()
        self.pattern_agent = PatternAnalyzer()
        self.anomaly_agent = AnomalyDetector()
        self.guardian_agent = GuardianAgent()
        self.consensus_engine = TrustWeightedConsensus()
        
        # Storage for results
        self.results = []
        
    def evaluate_full_dataset(self, ground_truth_file='data/synthetic_ehr_full.csv'):
        """
        Run evaluation on complete dataset with ground truth labels
        """
        print(f"{'='*80}")
        print(f"Starting Full Dataset Evaluation")
        print(f"{'='*80}\n")
        
        # Load ground truth data
        df_full = pd.read_csv(ground_truth_file)
        print(f"✓ Loaded {len(df_full)} records for {df_full['patient_id'].nunique()} patients")
        
        # Fit population statistics for anomaly detector
        print("✓ Fitting population statistics...")
        self.anomaly_agent.fit_population_statistics(df_full)
        
        # Process each patient
        patients = df_full['patient_id'].unique()
        print(f"✓ Processing {len(patients)} patients...\n")
        
        for idx, patient_id in enumerate(patients, 1):
            if idx % 10 == 0:
                print(f"  Progress: {idx}/{len(patients)} patients processed")
            
            patient_data = df_full[df_full['patient_id'] == patient_id].sort_values('record_date').reset_index(drop=True)
            
            # Run all agents
            trend_results = self.trend_agent.validate_patient_timeline(patient_data)
            pattern_results = self.pattern_agent.validate_patient_timeline(patient_data)
            anomaly_results = self.anomaly_agent.validate_patient_timeline(patient_data)
            
            # Process each record
            for record_idx, record in patient_data.iterrows():
                record_date = record['record_date']
                
                # Get agent results for this record
                trend_result = next((r for r in trend_results if r['record_date'] == record_date), {})
                pattern_result = next((r for r in pattern_results if r['record_date'] == record_date), {})
                anomaly_result = next((r for r in anomaly_results if r['record_date'] == record_date), {})
                
                agent_results = {
                    'trend': trend_result,
                    'pattern': pattern_result,
                    'anomaly': anomaly_result
                }
                
                # Guardian verification
                guardian_result = self.guardian_agent.verify_agents_consensus(record, agent_results)
                agent_results['guardian'] = guardian_result
                
                # Consensus decision
                consensus = self.consensus_engine.compute_consensus(agent_results, record)
                
                # Store result with ground truth
                self.results.append({
                    'patient_id': record['patient_id'],
                    'record_date': record['record_date'],
                    'ground_truth': record['has_error'],  # True = error exists
                    'error_type': record['error_type'],
                    
                    # Agent predictions (False = valid, True = flagged as invalid)
                    'trend_flagged': not trend_result.get('valid', True),
                    'pattern_flagged': not pattern_result.get('valid', True),
                    'anomaly_flagged': not anomaly_result.get('valid', True),
                    'consensus_flagged': consensus['final_decision'] == 'INVALID',
                    
                    # Confidences
                    'trend_confidence': trend_result.get('confidence', 0),
                    'pattern_confidence': pattern_result.get('confidence', 0),
                    'anomaly_confidence': anomaly_result.get('confidence', 0),
                    'consensus_confidence': consensus['confidence']
                })
        
        print(f"\n✓ Evaluation complete: {len(self.results)} records processed\n")
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(self.results)
        
        return self.results_df
    
    def calculate_metrics(self):
        """
        Calculate precision, recall, F1-score for each agent and consensus
        """
        if not hasattr(self, 'results_df'):
            raise ValueError("Must run evaluate_full_dataset() first!")
        
        print(f"{'='*80}")
        print(f"EVALUATION METRICS")
        print(f"{'='*80}\n")
        
        metrics = {}
        
        # Ground truth
        y_true = self.results_df['ground_truth'].astype(int)
        
        # Evaluate each agent + consensus
        agents = {
            'Trend Validator': 'trend_flagged',
            'Pattern Analyzer': 'pattern_flagged',
            'Anomaly Detector': 'anomaly_flagged',
            'Consensus': 'consensus_flagged'
        }
        
        for agent_name, column in agents.items():
            y_pred = self.results_df[column].astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            metrics[agent_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            }
            
            print(f"{agent_name}:")
            print(f"  Precision: {precision:.3f} (of flagged records, {precision*100:.1f}% were actual errors)")
            print(f"  Recall:    {recall:.3f} (detected {recall*100:.1f}% of actual errors)")
            print(f"  F1-Score:  {f1:.3f}")
            print(f"  Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}% correct)")
            print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}\n")
        
        self.metrics = metrics
        return metrics
    
    def plot_confusion_matrices(self, save_path='results/confusion_matrices.png'):
        """
        Plot confusion matrices for all agents
        """
        import os
        os.makedirs('results', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Confusion Matrices - Multi-Agent EHR Validation', fontsize=16, fontweight='bold')
        
        agents = ['trend_flagged', 'pattern_flagged', 'anomaly_flagged', 'consensus_flagged']
        titles = ['Trend Validator', 'Pattern Analyzer', 'Anomaly Detector', 'Consensus']
        
        y_true = self.results_df['ground_truth'].astype(int)
        
        for idx, (agent_col, title) in enumerate(zip(agents, titles)):
            ax = axes[idx // 2, idx % 2]
            y_pred = self.results_df[agent_col].astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       xticklabels=['Valid', 'Invalid'], 
                       yticklabels=['Valid', 'Invalid'])
            
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('Ground Truth')
            ax.set_xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrices saved to {save_path}")
        plt.close()
    
    def plot_performance_comparison(self, save_path='results/performance_comparison.png'):
        """
        Bar chart comparing agent performance
        """
        import os
        os.makedirs('results', exist_ok=True)
        
        # Extract metrics
        agents = list(self.metrics.keys())
        precision = [self.metrics[a]['precision'] for a in agents]
        recall = [self.metrics[a]['recall'] for a in agents]
        f1 = [self.metrics[a]['f1_score'] for a in agents]
        
        x = np.arange(len(agents))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
        ax.bar(x, recall, width, label='Recall', color='#3498db')
        ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
        
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Agent Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', fontsize=9)
            ax.text(i, r + 0.02, f'{r:.2f}', ha='center', fontsize=9)
            ax.text(i + width, f + 0.02, f'{f:.2f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Performance comparison saved to {save_path}")
        plt.close()
    
    def analyze_by_error_type(self):
        """
        Breakdown of detection rate by error type
        """
        print(f"\n{'='*80}")
        print(f"ERROR TYPE ANALYSIS")
        print(f"{'='*80}\n")
        
        # Filter only records with errors
        errors_df = self.results_df[self.results_df['ground_truth'] == True]
        
        error_types = errors_df['error_type'].unique()
        
        for error_type in sorted(error_types):
            if pd.isna(error_type):
                continue
            
            subset = errors_df[errors_df['error_type'] == error_type]
            
            trend_detected = subset['trend_flagged'].sum()
            pattern_detected = subset['pattern_flagged'].sum()
            anomaly_detected = subset['anomaly_flagged'].sum()
            consensus_detected = subset['consensus_flagged'].sum()
            total = len(subset)
            
            print(f"{error_type}:")
            print(f"  Total instances: {total}")
            print(f"  Trend:     {trend_detected}/{total} ({trend_detected/total*100:.1f}%)")
            print(f"  Pattern:   {pattern_detected}/{total} ({pattern_detected/total*100:.1f}%)")
            print(f"  Anomaly:   {anomaly_detected}/{total} ({anomaly_detected/total*100:.1f}%)")
            print(f"  Consensus: {consensus_detected}/{total} ({consensus_detected/total*100:.1f}%)\n")


# Run evaluation
if __name__ == "__main__":
    evaluator = EHRValidationEvaluator()
    
    # Step 1: Run evaluation
    results_df = evaluator.evaluate_full_dataset()
    
    # Step 2: Calculate metrics
    metrics = evaluator.calculate_metrics()
    
    # Step 3: Visualizations
    evaluator.plot_confusion_matrices()
    evaluator.plot_performance_comparison()
    
    # Step 4: Error type analysis
    evaluator.analyze_by_error_type()
    
    # Save results
    results_df.to_csv('results/evaluation_results.csv', index=False)
    print(f"\n✓ Full results saved to results/evaluation_results.csv")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")