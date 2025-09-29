"""
AutoDS - Enterprise Analytics Platform
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import re
from typing import Optional, Dict, List, Tuple

from config import AppConfig, AutoMLConfig, LIMEConfig, PreprocessingConfig
from pipeline import MLPipeline

# Constants
MAX_FILE_SIZE_MB = 200
CHART_DPI = 100
DEFAULT_CHART_HEIGHT = 5

# Page configuration
st.set_page_config(
    page_title="AutoDS", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Initialize session state
for key, default in [
    ('analysis_complete', False),
    ('results', None),
    ('df', None),
    ('target', None),
    ('pipeline', None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Professional color scheme
COLORS = {
    'primary': '#2E3440',      # Dark slate
    'secondary': '#4C566A',    # Medium grey
    'accent': '#5E81AC',       # Muted blue
    'success': '#A3BE8C',      # Sage green
    'warning': '#EBCB8B',      # Warm tan
    'error': '#BF616A',        # Muted red
    'bg_main': '#ECEFF4',      # Light grey
    'bg_card': '#FFFFFF',      # White
    'text_primary': '#2E3440', # Dark slate
    'text_secondary': '#4C566A', # Medium grey
    'border': '#D8DEE9'        # Light border
}

# Professional CSS
st.markdown(f"""
<style>
    /* Hide Streamlit header and menu */
    header {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Main layout */
    .main {{
        background-color: {COLORS['bg_main']};
        padding-top: 0.5rem;
    }}
    
    /* Typography */
    h1 {{
        color: {COLORS['text_primary']};
        font-weight: 600;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }}
    h2 {{
        color: {COLORS['text_primary']};
        font-weight: 600;
        font-size: 1.6rem;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }}
    h3 {{
        color: {COLORS['text_secondary']};
        font-weight: 500;
        font-size: 1.2rem;
        margin-bottom: 0.75rem;
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: {COLORS['primary']};
        color: white;
        border-radius: 4px;
        padding: 0.65rem 2rem;
        border: none;
        font-weight: 500;
        font-size: 1rem;
        letter-spacing: 0.01em;
        transition: all 0.2s;
    }}
    .stButton>button:hover {{
        background-color: {COLORS['secondary']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    
    /* Cards */
    .metric-card {{
        background-color: {COLORS['bg_card']};
        padding: 1.25rem;
        border-radius: 4px;
        border: 1px solid {COLORS['border']};
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .metric-card h3 {{
        color: {COLORS['text_secondary']};
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }}
    .metric-card .value {{
        color: {COLORS['text_primary']};
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1;
    }}
    
    /* Insight cards */
    .insight-card {{
        background-color: {COLORS['bg_card']};
        padding: 1.25rem;
        border-radius: 4px;
        border: 1px solid {COLORS['border']};
        margin: 0.75rem 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .insight-card h4 {{
        color: {COLORS['text_primary']};
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }}
    .insight-card p {{
        color: {COLORS['text_secondary']};
        line-height: 1.6;
        margin: 0.25rem 0;
        font-size: 0.95rem;
    }}
    
    /* Summary box */
    .summary-box {{
        background-color: {COLORS['bg_card']};
        padding: 1.75rem;
        border-radius: 4px;
        border-left: 3px solid {COLORS['primary']};
        margin: 1.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .summary-box p {{
        color: {COLORS['text_primary']};
        font-size: 1.05rem;
        line-height: 1.7;
        margin: 0;
    }}
    
    /* Action items */
    .action-item {{
        background-color: {COLORS['bg_card']};
        padding: 1.15rem;
        border-radius: 4px;
        border-left: 3px solid {COLORS['secondary']};
        margin: 0.75rem 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .action-item.high {{border-left-color: {COLORS['error']}; }}
    .action-item.medium {{border-left-color: {COLORS['warning']}; }}
    .action-item.low {{border-left-color: {COLORS['success']}; }}
    .action-item h4 {{
        color: {COLORS['text_primary']};
        font-size: 1rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }}
    .action-item p {{
        color: {COLORS['text_secondary']};
        line-height: 1.6;
        margin: 0.25rem 0;
        font-size: 0.95rem;
    }}
    .priority-badge {{
        color: {COLORS['text_secondary']};
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    /* Status indicators */
    .status-good {{
        background-color: #f0f7f4;
        color: #1e5a3a;
        padding: 1rem;
        border-radius: 4px;
        border-left: 3px solid {COLORS['success']};
    }}
    .status-warning {{
        background-color: #fefaf0;
        color: #6b5416;
        padding: 1rem;
        border-radius: 4px;
        border-left: 3px solid {COLORS['warning']};
    }}
    
    /* Sidebar */
    .css-1d391kg {{
        background-color: {COLORS['bg_card']};
    }}
</style>
""", unsafe_allow_html=True)


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Validate uploaded dataframe"""
    if df.empty:
        return False, "File is empty"
    if len(df.columns) < 2:
        return False, "Need at least 2 columns (features and target)"
    if len(df) < 10:
        return False, "Need at least 10 rows for meaningful analysis"
    return True, None


def clean_feature_name(name: str) -> Optional[str]:
    """Filter out engineered or invalid feature names"""
    if re.match(r'^-?\d+\.?\d*$', str(name)):
        return None
    for suffix in ['_year', '_month', '_day', '_dayofweek', '_quarter']:
        if str(name).endswith(suffix):
            return None
    return name


def aggregate_feature_importance(
    importance: Dict[str, float], 
    original_cols: List[str]
) -> Dict[str, float]:
    """Aggregate engineered features back to source columns"""
    aggregated = {}
    
    for feat, imp in importance.items():
        if not clean_feature_name(feat):
            continue
            
        base = feat
        for orig in original_cols:
            if feat.startswith(orig) or feat == orig:
                base = orig
                break
        
        aggregated[base] = aggregated.get(base, 0) + imp
    
    return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))


def analyze_data_quality(df: pd.DataFrame) -> List[Dict]:
    """Comprehensive data quality analysis"""
    issues = []
    
    # Missing data analysis
    missing = df.isnull().sum()
    critical_missing = missing[missing > len(df) * 0.3]
    if len(critical_missing) > 0:
        issues.append({
            'severity': 'high',
            'message': f'{len(critical_missing)} columns with >30% missing data',
            'action': 'Consider removing: ' + ', '.join(critical_missing.index[:3].tolist()),
            'columns': critical_missing.index.tolist()
        })
    
    # Duplicate detection
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append({
            'severity': 'medium',
            'message': f'{dup_count} duplicate rows ({dup_count/len(df)*100:.1f}%)',
            'action': 'Remove duplicates to improve model accuracy',
            'columns': []
        })
    
    # Low variance check
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if len(df[col].dropna()) > 0:
            std = df[col].std()
            mean = df[col].mean()
            if abs(mean) > 1e-10 and std / abs(mean) < 0.01:
                issues.append({
                    'severity': 'low',
                    'message': f'{col} has very low variance',
                    'action': 'May not contribute to predictions',
                    'columns': [col]
                })
    
    return issues


def generate_strategic_recommendations(
    target: str, 
    top_features: List[str], 
    perf: Dict,
    df: pd.DataFrame
) -> List[Dict]:
    """Generate actionable business recommendations"""
    recommendations = []
    
    # Primary drivers recommendation
    if top_features:
        recommendations.append({
            'priority': 'HIGH',
            'title': 'Focus on Key Drivers',
            'description': f"Priority areas: {', '.join(top_features[:3])}",
            'impact': f"These {len(top_features[:3])} factors explain the majority of variation in {target}. " +
                     "Resource allocation to these areas will yield maximum ROI."
        })
    
    # Model quality assessment
    if 'rmse' in perf:
        error = perf['rmse']
        if error < 0.1:
            recommendations.append({
                'priority': 'MEDIUM',
                'title': 'Deploy Model to Production',
                'description': f'Model achieves strong predictive accuracy (error: {error:.3f})',
                'impact': 'Integrate predictions into decision workflows for data-driven operations'
            })
        elif error > 0.3:
            recommendations.append({
                'priority': 'HIGH',
                'title': 'Improve Data Collection',
                'description': 'Current model accuracy is limited by available data',
                'impact': 'Collect additional features or higher quality data to improve predictions'
            })
    
    # Data volume assessment
    if len(df) < 100:
        recommendations.append({
            'priority': 'HIGH',
            'title': 'Increase Sample Size',
            'description': f'Current dataset has only {len(df)} records',
            'impact': 'Collect more data to improve model reliability and reduce overfitting risk'
        })
    
    return recommendations


def create_driver_chart(
    aggregated: Dict[str, float], 
    num_features: int
) -> plt.Figure:
    """Create professional feature importance chart with blue gradient"""
    top_n = min(num_features, len(aggregated))
    items = list(aggregated.items())[:top_n]
    features, values = zip(*items)
    
    fig, ax = plt.subplots(figsize=(10, max(DEFAULT_CHART_HEIGHT, len(features) * 0.4)))
    fig.patch.set_facecolor(COLORS['bg_main'])
    ax.set_facecolor(COLORS['bg_card'])
    
    # Blue gradient based on importance
    colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(features)))
    bars = ax.barh(range(len(features)), values, color=colors, edgecolor='white', linewidth=1.5)
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10, color=COLORS['text_primary'])
    ax.invert_yaxis()
    ax.set_xlabel('Relative Impact', fontsize=11, color=COLORS['text_secondary'], fontweight=500)
    
    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['border'])
    ax.spines['bottom'].set_color(COLORS['border'])
    ax.grid(axis='x', alpha=0.2, linestyle='-', linewidth=0.5, color=COLORS['border'])
    ax.tick_params(colors=COLORS['text_secondary'])
    
    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val, bar.get_y() + bar.get_height()/2, f'  {val:.2f}', 
               va='center', fontsize=9, color=COLORS['text_primary'], fontweight=500)
    
    plt.tight_layout()
    return fig


def create_distribution_charts(
    df: pd.DataFrame, 
    target: str, 
    top_features: List[str]
) -> Optional[plt.Figure]:
    """Create distribution analysis charts with blue styling"""
    numeric_cols = [c for c in top_features if c in df.select_dtypes(include=[np.number]).columns][:4]
    
    if not numeric_cols:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=CHART_DPI)
    fig.patch.set_facecolor(COLORS['bg_main'])
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        if idx < 4:
            ax = axes[idx]
            ax.hist(df[col].dropna(), bins=20, color='#5E81AC', 
                   alpha=0.75, edgecolor='white', linewidth=1.2)
            ax.set_title(col, fontweight=600, fontsize=11, color=COLORS['text_primary'])
            ax.set_xlabel('Value', fontsize=9, color=COLORS['text_secondary'])
            ax.set_ylabel('Frequency', fontsize=9, color=COLORS['text_secondary'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(COLORS['border'])
            ax.spines['bottom'].set_color(COLORS['border'])
            ax.set_facecolor(COLORS['bg_card'])
            ax.grid(axis='y', alpha=0.2, color=COLORS['border'])
            ax.tick_params(colors=COLORS['text_secondary'])
    
    for idx in range(len(numeric_cols), 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def create_correlation_heatmap(
    df: pd.DataFrame, 
    target: str, 
    top_features: List[str]
) -> Optional[plt.Figure]:
    """Create correlation heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    relevant_cols = [c for c in top_features if c in numeric_cols]
    
    if target in numeric_cols and target not in relevant_cols:
        relevant_cols.append(target)
    
    if len(relevant_cols) < 2:
        return None
    
    corr_matrix = df[relevant_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=CHART_DPI)
    fig.patch.set_facecolor(COLORS['bg_main'])
    ax.set_facecolor(COLORS['bg_card'])
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(relevant_cols)))
    ax.set_yticks(np.arange(len(relevant_cols)))
    ax.set_xticklabels(relevant_cols, rotation=45, ha='right', color=COLORS['text_primary'])
    ax.set_yticklabels(relevant_cols, color=COLORS['text_primary'])
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, 
                   color=COLORS['text_secondary'])
    cbar.ax.tick_params(colors=COLORS['text_secondary'])
    
    # Add correlation values
    for i in range(len(relevant_cols)):
        for j in range(len(relevant_cols)):
            val = corr_matrix.iloc[i, j]
            text_color = "white" if abs(val) > 0.5 else COLORS['text_primary']
            ax.text(j, i, f'{val:.2f}', ha="center", va="center", 
                   color=text_color, fontsize=9, fontweight=500)
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight=600, 
                pad=20, color=COLORS['text_primary'])
    plt.tight_layout()
    return fig


# Sidebar
with st.sidebar:
    st.markdown("### Configuration")
    max_models = st.slider("Model Iterations", 5, 30, 15, 
                          help="More iterations = better accuracy but longer training time")
    max_secs = st.slider("Max Training Time (min)", 1, 15, 5) * 60
    num_features = st.slider("Features to Analyze", 5, 15, 8)
    
    st.markdown("---")
    
    if st.session_state.analysis_complete:
        if st.button("New Analysis"):
            for key in ['analysis_complete', 'results', 'df', 'target', 'pipeline']:
                st.session_state[key] = None if key != 'analysis_complete' else False
            st.rerun()


# Main content
st.title("AutoDS")
st.caption("Enterprise Analytics Platform")

if not st.session_state.analysis_complete:
    # UPLOAD INTERFACE
    st.markdown("### Data Upload")
    
    uploaded_file = st.file_uploader(
        "",
        type=['csv', 'xlsx', 'xls'],
        label_visibility="collapsed",
        help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
    )
    
    if uploaded_file:
        # File size check
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File too large ({file_size_mb:.1f}MB). Maximum size is {MAX_FILE_SIZE_MB}MB.")
            st.stop()
        
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validate
            valid, error_msg = validate_dataframe(df)
            if not valid:
                st.error(f"Invalid data: {error_msg}")
                st.stop()
            
            st.session_state.df = df
            
            # Dataset overview
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = [
                ("Records", f"{len(df):,}"),
                ("Features", str(len(df.columns))),
                ("Complete", f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.0f}%"),
                ("Size", f"{file_size_mb:.1f} MB")
            ]
            
            for col, (label, value) in zip([col1, col2, col3, col4], metrics):
                with col:
                    st.markdown(f'<div class="metric-card"><h3>{label}</h3><p class="value">{value}</p></div>', 
                               unsafe_allow_html=True)
            
            st.markdown("")
            
            with st.expander("Preview Data"):
                st.dataframe(df.head(20), use_container_width=True)
            
            st.markdown("---")
            st.markdown("### Analysis Configuration")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                target = st.selectbox("Target Variable", options=df.columns.tolist())
                st.session_state.target = target
            with col2:
                exclude_cols = st.multiselect(
                    "Exclude Columns (IDs, irrelevant fields)",
                    options=[c for c in df.columns if c != target]
                )
            
            st.markdown("")
            
            if st.button("Run Analysis", type="primary"):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    df.to_csv(f.name, index=False)
                    temp_file = f.name
                
                try:
                    config = AppConfig(
                        automl=AutoMLConfig(max_models=max_models, max_runtime_secs=max_secs),
                        lime=LIMEConfig(num_features=num_features, num_samples=10),
                        preprocessing=PreprocessingConfig(scale_numeric=True, engineer_features=True)
                    )
                    
                    progress_bar = st.progress(0)
                    status = st.empty()
                    
                    status.text("Preparing dataset...")
                    progress_bar.progress(20)
                    
                    pipeline = MLPipeline(config)
                    pipeline.load_data(temp_file)
                    pipeline.preprocess_data(target)
                    
                    status.text("Training predictive models...")
                    progress_bar.progress(50)
                    pipeline.train_model(target, exclude_cols if exclude_cols else None)
                    
                    status.text("Generating explanations...")
                    progress_bar.progress(80)
                    pipeline.explain_model(target)
                    
                    progress_bar.progress(100)
                    status.empty()
                    
                    st.session_state.results = pipeline.results
                    st.session_state.pipeline = pipeline
                    st.session_state.analysis_complete = True
                    
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    with st.expander("Error Details"):
                        st.exception(e)
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
        
        except Exception as e:
            st.error(f"Unable to load file: {str(e)}")
    else:
        st.info("Upload a CSV or Excel file to begin analysis")

else:
    # RESULTS INTERFACE
    results = st.session_state.results
    df = st.session_state.df
    target = st.session_state.target
    
    perf = results.get('model_performance', {})
    importance = results.get('feature_importance', {})
    
    # Process feature importance
    original_cols = [c for c in df.columns if c != target]
    aggregated = aggregate_feature_importance(importance, original_cols)
    top_features = list(aggregated.keys())[:8]
    
    # EXECUTIVE SUMMARY
    st.header("Executive Summary")
    
    summary_parts = []
    if 'rmse' in perf:
        error = perf['rmse']
        quality = "high" if error < 0.1 else "moderate" if error < 0.3 else "limited"
        summary_parts.append(f"Model achieves {quality} predictive accuracy (error: {error:.3f})")
    
    if aggregated:
        top_driver = list(aggregated.keys())[0]
        summary_parts.append(f"{top_driver} is the primary driver of {target}")
    
    model_type = perf.get('model_type', 'unknown').upper()
    summary_parts.append(f"Optimal algorithm: {model_type}")
    
    st.markdown(f'''
    <div class="summary-box">
    <p>{'. '.join(summary_parts)}.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # KEY FINDINGS
    st.markdown("")
    st.header("Key Findings")
    
    # Use exact height matching for columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("") # No extra space needed
        if aggregated:
            fig = create_driver_chart(aggregated, num_features)
            st.pyplot(fig)
            plt.close()
    
    with col2:
        st.markdown("") # Align with col1
        for i, (feat, imp) in enumerate(list(aggregated.items())[:3], 1):
            relative = imp / max(aggregated.values())
            level = "Critical" if relative > 0.7 else ("High" if relative > 0.4 else "Moderate")
            
            st.markdown(f'''
            <div class="insight-card">
            <h4>{i}. {feat}</h4>
            <p><strong>{level} Impact</strong><br>
            Key driver of {target} outcomes</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # STRATEGIC RECOMMENDATIONS
    st.markdown("")
    st.header("Strategic Recommendations")
    
    recommendations = generate_strategic_recommendations(target, top_features, perf, df)
    
    for rec in recommendations:
        priority_class = rec['priority'].lower()
        st.markdown(f'''
        <div class="action-item {priority_class}">
        <p class="priority-badge">{rec['priority']} PRIORITY</p>
        <h4>{rec['title']}</h4>
        <p>{rec['description']}</p>
        <p><strong>Business Impact:</strong> {rec['impact']}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # DATA QUALITY
    st.markdown("")
    st.header("Data Quality Assessment")
    
    quality_issues = analyze_data_quality(df)
    
    if not quality_issues:
        st.markdown('<div class="status-good"><strong>Data Quality: Excellent</strong><br>No significant issues detected</div>', 
                   unsafe_allow_html=True)
    else:
        high_priority = [i for i in quality_issues if i['severity'] == 'high']
        other_issues = [i for i in quality_issues if i['severity'] != 'high']
        
        for issue in high_priority:
            st.markdown(f'''
            <div class="status-warning">
            <strong>{issue["message"]}</strong><br>
            Recommended action: {issue["action"]}
            </div>
            ''', unsafe_allow_html=True)
        
        if other_issues:
            with st.expander(f"View {len(other_issues)} Additional Observations"):
                for issue in other_issues:
                    st.markdown(f"**{issue['message']}** - {issue['action']}")
    
    # DETAILED ANALYSIS
    st.markdown("")
    st.header("Detailed Analysis")
    
    tab1, tab2 = st.tabs(["Distributions", "Correlations"])
    
    with tab1:
        dist_fig = create_distribution_charts(df, target, top_features)
        if dist_fig:
            st.pyplot(dist_fig)
            plt.close()
            st.caption("Distribution patterns reveal data characteristics and potential outliers")
        else:
            st.info("Distribution analysis requires numeric variables")
    
    with tab2:
        corr_fig = create_correlation_heatmap(df, target, top_features)
        if corr_fig:
            st.pyplot(corr_fig)
            plt.close()
            st.caption("Correlation matrix shows relationships between variables. Values near +1/-1 indicate strong relationships.")
        else:
            st.info("Correlation analysis requires numeric variables")
    
    # MODEL DETAILS
    with st.expander("View Model Performance Details"):
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame([
            {'Metric': k.upper(), 'Value': f'{v:.4f}' if isinstance(v, (int, float)) else str(v)}
            for k, v in perf.items()
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        if 'leaderboard' in results:
            st.subheader("Model Leaderboard")
            st.dataframe(results['leaderboard'].head(10), use_container_width=True)
    
    # EXPORT
    st.markdown("---")
    st.markdown("### Export Results")
    
    # Prepare downloads
    data_csv = df.to_csv(index=False).encode('utf-8')
    models_csv = results['leaderboard'].to_csv(index=False).encode('utf-8') if 'leaderboard' in results else None
    
    drivers_csv = None
    if aggregated:
        importance_df = pd.DataFrame([{'Feature': k, 'Impact': v} for k, v in aggregated.items()])
        drivers_csv = importance_df.to_csv(index=False).encode('utf-8')
    
    summary_lines = [
        f"ANALYTICS REPORT",
        f"Target Variable: {target}",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "EXECUTIVE SUMMARY:",
        *[f"  {line}" for line in summary_parts],
        "",
        "TOP DRIVERS:",
        *[f"  {i}. {feat} (impact: {imp:.3f})" for i, (feat, imp) in enumerate(list(aggregated.items())[:5], 1)],
        "",
        "MODEL PERFORMANCE:",
        *[f"  {k.upper()}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k.upper()}: {v}" 
          for k, v in perf.items()]
    ]
    report_txt = "\n".join(summary_lines).encode('utf-8')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.download_button("Data", data_csv, "data.csv", "text/csv", use_container_width=True)
    with col2:
        if models_csv:
            st.download_button("Models", models_csv, "models.csv", "text/csv", use_container_width=True)
    with col3:
        if drivers_csv:
            st.download_button("Drivers", drivers_csv, "drivers.csv", "text/csv", use_container_width=True)
    with col4:
        st.download_button("Report", report_txt, "report.txt", "text/plain", use_container_width=True)

st.markdown("---")
st.caption("AutoDS Enterprise Analytics Platform • Powered by H2O AutoML")