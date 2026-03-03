import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.trend_validator import TrendValidator
from agents.pattern_analyzer import PatternAnalyzer
from agents.anomaly_detector import AnomalyDetector
from agents.guardian_agent import GuardianAgent
from core.consensus import TrustWeightedConsensus
from core.explanation_engine import ExplanationEngine

# Page config
st.set_page_config(
    page_title="EADC - Temporal EHR Validator",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
    st.session_state.df_all = None

# Title
st.title("🏥 Temporal EHR Validation System")
st.markdown("**Multi-Agent Framework for Explainable Temporal Consistency Validation**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Controls")
    
    # Load data button
    if st.button("🔄 Initialize System", type="primary"):
        with st.spinner("Loading data and initializing agents..."):
            try:
                # Load data
                st.session_state.df_all = pd.read_csv('data/synthetic_ehr_test.csv')
                
                # Initialize agents
                st.session_state.trend_agent = TrendValidator()
                st.session_state.pattern_agent = PatternAnalyzer()
                st.session_state.anomaly_agent = AnomalyDetector()
                st.session_state.anomaly_agent.fit_population_statistics(st.session_state.df_all)
                st.session_state.guardian_agent = GuardianAgent()
                st.session_state.consensus_engine = TrustWeightedConsensus()
                st.session_state.explanation_engine = ExplanationEngine()
                
                st.session_state.agents_initialized = True
                st.success("✓ System initialized!")
            except Exception as e:
                st.error(f"Error: {e}")
    
    if st.session_state.agents_initialized:
        st.success("✓ System Ready")
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.metric("Total Records", len(st.session_state.df_all))
        st.metric("Patients", st.session_state.df_all['patient_id'].nunique())
    
    st.markdown("---")
    st.markdown("### 🤖 Agents")
    st.markdown("1. Trend Validator")
    st.markdown("2. Pattern Analyzer")
    st.markdown("3. Anomaly Detector")
    st.markdown("4. Guardian Agent")

# Main area
if not st.session_state.agents_initialized:
    st.info(" Click **Initialize System** in the sidebar to begin")
    
    # Show architecture diagram
    st.markdown("### 🏗️ System Architecture")
    st.image("Architecture.png", use_container_width=True)
    
else:
    # Patient selection
    patients = sorted(st.session_state.df_all['patient_id'].unique())
    selected_patient = st.selectbox("Select Patient", patients, index=0)
    
    # Get patient data
    patient_data = st.session_state.df_all[
        st.session_state.df_all['patient_id'] == selected_patient
    ].sort_values('record_date').reset_index(drop=True)
    
    # Show patient timeline
    st.markdown(f"### 📅 Timeline for {selected_patient}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(patient_data))
    with col2:
        st.metric("Date Range", f"{patient_data['record_date'].min()} to {patient_data['record_date'].max()}")
    with col3:
        st.metric("Time Span", f"{len(patient_data) * 15} days")
    
    # Process button
    if st.button("🔍 Validate Timeline", type="primary"):
        with st.spinner("Running validation..."):
            # Run all agents
            trend_results = st.session_state.trend_agent.validate_patient_timeline(patient_data)
            pattern_results = st.session_state.pattern_agent.validate_patient_timeline(patient_data)
            anomaly_results = st.session_state.anomaly_agent.validate_patient_timeline(patient_data)
            
            # Store results
            st.session_state.validation_results = {
                'trend': trend_results,
                'pattern': pattern_results,
                'anomaly': anomaly_results,
                'patient_data': patient_data
            }
            
            st.success("✓ Validation complete!")
    
    # Display results
    if 'validation_results' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Validation Results")
        
        results = st.session_state.validation_results
        patient_data = results['patient_data']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Count flags
        trend_flags = sum(1 for r in results['trend'] if not r['valid'])
        pattern_flags = sum(1 for r in results['pattern'] if not r['valid'])
        anomaly_flags = sum(1 for r in results['anomaly'] if not r['valid'])
        
        with col1:
            st.metric("Trend Flags", trend_flags)
        with col2:
            st.metric("Pattern Flags", pattern_flags)
        with col3:
            st.metric("Anomaly Flags", anomaly_flags)
        with col4:
            total_flags = len(set(
                [r['record_date'] for r in results['trend'] if not r['valid']] +
                [r['record_date'] for r in results['pattern'] if not r['valid']] +
                [r['record_date'] for r in results['anomaly'] if not r['valid']]
            ))
            st.metric("Unique Records Flagged", total_flags)
        
        # Record-by-record display
        st.markdown("### 📝 Record Details")
        
        for idx, record in patient_data.iterrows():
            record_date = record['record_date']
            
            # Get results for this record
            trend_result = next((r for r in results['trend'] if r['record_date'] == record_date), {})
            pattern_result = next((r for r in results['pattern'] if r['record_date'] == record_date), {})
            anomaly_result = next((r for r in results['anomaly'] if r['record_date'] == record_date), {})
            
            agent_results = {
                'trend': trend_result,
                'pattern': pattern_result,
                'anomaly': anomaly_result
            }
            
            # Guardian verification
            guardian_result = st.session_state.guardian_agent.verify_agents_consensus(record, agent_results)
            agent_results['guardian'] = guardian_result
            
            # Consensus
            consensus = st.session_state.consensus_engine.compute_consensus(agent_results, record)
            
            # Generate explanation
            explanation = st.session_state.explanation_engine.generate_explanation(
                consensus, agent_results, record
            )
            
            # Display in expander
            status_emoji = "✅" if consensus['final_decision'] == 'VALID' else "❌"
            confidence_pct = f"{consensus['confidence']*100:.0f}%"
            
            with st.expander(f"{status_emoji} {record_date} - {consensus['final_decision']} (Confidence: {confidence_pct})"):
                # Record data
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**📊 Vitals:**")
                    st.write(f"Weight: **{record['weight_kg']}kg**")
                    st.write(f"Heart Rate: **{record['heart_rate_bpm']}bpm**")
                    st.write(f"BP: **{record['systolic_bp_mmhg']}/{record['diastolic_bp_mmhg']}mmHg**")
                    st.write(f"Temperature: **{record['temperature_celsius']}°C**")
                
                with col2:
                    st.markdown("**🤖 Agent Decisions:**")
                    for agent_name in ['trend', 'pattern', 'anomaly']:
                        result = agent_results[agent_name]
                        decision = "✓ VALID" if result.get('valid', True) else "✗ INVALID"
                        conf = result.get('confidence', 0)
                        st.write(f"{agent_name.capitalize()}: {decision} ({conf:.0%})")
                
                # Tabs for explanation layers
                tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🧠 Reasoning", "📊 Evidence", "💡 Recommendation"])
                
                with tab1:
                    st.markdown(explanation['layer1_summary'])
                
                with tab2:
                    for line in explanation['layer2_reasoning']:
                        st.markdown(line)
                
                with tab3:
                    evidence = explanation['layer3_evidence']
                    
                    st.markdown("**Record Details:**")
                    st.json(evidence['record_details'])
                    
                    st.markdown("**Agent Evidence:**")
                    for agent_name, agent_ev in evidence['agent_evidence'].items():
                        with st.expander(f"{agent_name.capitalize()} Evidence"):
                            st.json(agent_ev)
                    
                    st.markdown("**Guardian Verification:**")
                    st.json(evidence['guardian_verification'])
                
                with tab4:
                    rec = explanation['recommendation']
                    
                    if rec['action'] == 'REJECT':
                        st.error(f"**Action:** {rec['action']}")
                    elif rec['action'] == 'HUMAN_REVIEW':
                        st.warning(f"**Action:** {rec['action']}")
                    elif rec['action'] == 'FLAG':
                        st.warning(f"**Action:** {rec['action']}")
                    else:
                        st.success(f"**Action:** {rec['action']}")
                    
                    st.markdown(f"**Reason:** {rec['reason']}")
                    st.markdown("**Next Steps:**")
                    for step in rec['next_steps']:
                        st.markdown(f"- {step}")

# Footer
st.markdown("---")
st.markdown("**EADC - Explainable Agentic Data Curator** | Multi-Agent Temporal Validation System")