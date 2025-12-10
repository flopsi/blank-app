"""
Multi-Category Protein Classification for Proteomics Benchmarking
Bayesian-equivalence testing framework for 3-species mix experiments
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO

st.set_page_config(page_title="Proteomics Benchmark Classifier", layout="wide", page_icon="🧬")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_columns(df):
    """Auto-detect protein_id, species, and log2_fc columns."""
    cols_lower = {col.lower(): col for col in df.columns}
    
    # Detect protein ID column
    protein_col = None
    for key in ['protein_id', 'protein', 'id', 'gene', 'gene_name', 'accession']:
        if key in cols_lower:
            protein_col = cols_lower[key]
            break
    if not protein_col and len(df.columns) > 0:
        protein_col = df.columns[0]
    
    # Detect species column
    species_col = None
    for key in ['species', 'organism', 'org']:
        if key in cols_lower:
            species_col = cols_lower[key]
            break
    
    # Detect fold-change column
    fc_col = None
    for key in ['log2_fc', 'log2fc', 'logfc', 'fc', 'log2foldchange', 'fold_change']:
        if key in cols_lower:
            fc_col = cols_lower[key]
            break
    if not fc_col and len(df.columns) > 1:
        fc_col = df.columns[1]
    
    return protein_col, species_col, fc_col


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def tost_vs_expected(observed: np.ndarray, expected: float, tolerance: float, alpha: float = 0.05) -> dict:
    """TOST to test if observed values equal expected value within tolerance."""
    data = np.array(observed)
    n = len(data)
    if n < 2:
        return {'p_value': 1.0, 'equivalent': False, 'mean': np.mean(data), 
                'expected': expected, 'CI_90': (np.nan, np.nan), 'bounds': (expected - tolerance, expected + tolerance)}
    
    mean = np.mean(data)
    se = stats.sem(data)
    df = n - 1
    
    lower_bound = expected - tolerance
    upper_bound = expected + tolerance
    
    if se == 0:
        within = lower_bound <= mean <= upper_bound
        return {'p_value': 0.0 if within else 1.0, 'equivalent': within, 'mean': mean,
                'expected': expected, 'CI_90': (mean, mean), 'bounds': (lower_bound, upper_bound)}
    
    t_lower = (mean - lower_bound) / se
    t_upper = (mean - upper_bound) / se
    
    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    
    p_tost = max(p_lower, p_upper)
    ci_90 = stats.t.interval(0.90, df, loc=mean, scale=se)
    
    return {
        'p_value': p_tost,
        'equivalent': p_tost < alpha,
        'mean': mean,
        'expected': expected,
        'CI_90': ci_90,
        'bounds': (lower_bound, upper_bound)
    }


def classify_protein(posterior_samples: np.ndarray, expected_fc: float, equiv_tolerance: float, 
                     direction_threshold: float = 0.5, direction_prob_cutoff: float = 0.95) -> dict:
    """Classify protein into four categories based on posterior distribution."""
    posterior = np.array(posterior_samples)
    mean_fc = np.mean(posterior)
    ci_90 = np.percentile(posterior, [5, 95])
    
    if expected_fc == 0:  # Background species - should be unchanged
        prob_unchanged = np.mean(np.abs(posterior) < direction_threshold)
        prob_detected_change = 1 - prob_unchanged
        if prob_detected_change > direction_prob_cutoff:
            category = "TRUE_FALSE_POSITIVE"
        else:
            category = "CORRECT_UNCHANGED"
        return {'category': category, 'mean_fc': mean_fc, 'ci_90': ci_90, 
                'prob_correct_direction': prob_unchanged, 'expected_fc': expected_fc}
    
    # Calculate direction probability
    if expected_fc > 0:
        prob_correct_direction = np.mean(posterior > 0)
    else:
        prob_correct_direction = np.mean(posterior < 0)
    
    if prob_correct_direction < direction_prob_cutoff:
        category = "WRONG_DIRECTION"
    else:
        lower_bound = expected_fc - equiv_tolerance
        upper_bound = expected_fc + equiv_tolerance
        within_bounds = (ci_90[0] >= lower_bound) and (ci_90[1] <= upper_bound)
        
        if within_bounds:
            category = "CORRECT_DIRECTION_AND_MAGNITUDE"
        else:
            category = "CORRECT_DIRECTION_WRONG_MAGNITUDE"
    
    return {'category': category, 'mean_fc': mean_fc, 'ci_90': ci_90,
            'prob_correct_direction': prob_correct_direction, 'expected_fc': expected_fc}


def calculate_deFDR(classifications: dict, expected_directions: dict) -> float:
    """Calculate directional error FDR."""
    true_positives = 0
    false_positives = 0
    
    for protein_id, result in classifications.items():
        category = result['category']
        if category in ['CORRECT_DIRECTION_AND_MAGNITUDE', 'CORRECT_DIRECTION_WRONG_MAGNITUDE']:
            true_positives += 1
        elif category in ['WRONG_DIRECTION', 'TRUE_FALSE_POSITIVE']:
            false_positives += 1
    
    if true_positives + false_positives == 0:
        return 0.0
    return false_positives / (false_positives + true_positives)


def calculate_asymmetry_factor(log2_fold_changes: np.ndarray) -> float:
    """Calculate asymmetry factor from density function."""
    fc = np.array(log2_fold_changes)
    if len(fc) < 3:
        return np.nan
    
    try:
        kde = gaussian_kde(fc)
        x_range = np.linspace(fc.min() - 1, fc.max() + 1, 1000)
        density = kde(x_range)
        
        peak_idx = np.argmax(density)
        peak_x = x_range[peak_idx]
        max_height = density[peak_idx]
        threshold = 0.10 * max_height
        
        left_side = density[:peak_idx]
        right_side = density[peak_idx:]
        
        left_crossings = np.where(left_side < threshold)[0]
        left_x = x_range[left_crossings[-1]] if len(left_crossings) > 0 else x_range[0]
        
        right_crossings = np.where(right_side < threshold)[0]
        right_x = x_range[peak_idx + right_crossings[0]] if len(right_crossings) > 0 else x_range[-1]
        
        left_distance = abs(peak_x - left_x)
        right_distance = abs(right_x - peak_x)
        
        if right_distance == 0:
            return float('inf')
        return left_distance / right_distance
    except:
        return np.nan


def calculate_stratified_rmse(observed_fc: np.ndarray, expected_fc: np.ndarray, 
                               species_labels: np.ndarray) -> dict:
    """Calculate RMSE stratified by species."""
    results = {}
    unique_species = np.unique(species_labels)
    
    for species in unique_species:
        mask = species_labels == species
        obs = observed_fc[mask]
        exp = expected_fc[mask]
        
        if len(obs) > 0:
            rmse = np.sqrt(np.mean((obs - exp) ** 2))
            bias = np.mean(obs - exp)
            results[species] = {'RMSE': rmse, 'bias': bias, 'n': len(obs)}
    
    return results


def proteomics_capability_analysis(observed_fc: np.ndarray, expected_fc: float, 
                                    tolerance_pct: float = 0.25) -> dict:
    """Calculate process capability for proteomics fold-change data."""
    target = expected_fc
    tolerance = abs(target) * tolerance_pct if target != 0 else 0.5
    
    USL = target + tolerance
    LSL = target - tolerance
    
    data = np.array(observed_fc)
    if len(data) < 2:
        return {'Cp': np.nan, 'Cpk': np.nan, 'Cpm': np.nan, 'PPM_defective': np.nan, 'Within_spec_%': np.nan}
    
    mean = np.mean(data)
    sigma = np.std(data, ddof=1)
    
    if sigma == 0:
        sigma = 1e-10
    
    Cp = (USL - LSL) / (6 * sigma)
    Cpu = (USL - mean) / (3 * sigma)
    Cpl = (mean - LSL) / (3 * sigma)
    Cpk = min(Cpu, Cpl)
    Cpm = (USL - LSL) / (6 * np.sqrt(sigma**2 + (mean - target)**2))
    
    p_below = stats.norm.cdf(LSL, mean, sigma)
    p_above = 1 - stats.norm.cdf(USL, mean, sigma)
    ppm_out_of_spec = (p_below + p_above) * 1e6
    
    return {
        'Target': target, 'Mean': mean, 'Std_Dev': sigma,
        'USL': USL, 'LSL': LSL, 'Cp': Cp, 'Cpk': Cpk, 'Cpm': Cpm,
        'PPM_defective': ppm_out_of_spec, 'Within_spec_%': (1 - p_below - p_above) * 100
    }


def compare_fc_distributions(observed: np.ndarray, expected: np.ndarray) -> dict:
    """Comprehensive distribution comparison."""
    results = {}
    
    if len(observed) < 3 or len(expected) < 3:
        return {'KS': {'D': np.nan, 'p': np.nan}}
    
    ks = stats.ks_2samp(observed, expected)
    results['KS'] = {'D': ks.statistic, 'p': ks.pvalue}
    
    try:
        cvm = stats.cramervonmises_2samp(observed, expected)
        results['CvM'] = {'statistic': cvm.statistic, 'p': cvm.pvalue}
    except:
        results['CvM'] = {'statistic': np.nan, 'p': np.nan}
    
    return results


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🧬 Multi-Category Protein Classification")
st.markdown("**Bayesian-equivalence testing framework for proteomics benchmarking**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Expected Fold-Changes (log2)")
    species_1_name = st.text_input("Species 1 Name", value="Human")
    species_1_fc = st.number_input(f"{species_1_name} expected log2FC", value=0.0, step=0.1)
    
    species_2_name = st.text_input("Species 2 Name", value="Yeast")
    species_2_fc = st.number_input(f"{species_2_name} expected log2FC", value=1.0, step=0.1)
    
    species_3_name = st.text_input("Species 3 Name", value="E.coli")
    species_3_fc = st.number_input(f"{species_3_name} expected log2FC", value=-2.0, step=0.1)
    
    st.divider()
    st.subheader("Classification Parameters")
    equiv_tolerance_pct = st.slider("Equivalence tolerance (%)", 10, 50, 25) / 100
    direction_prob_cutoff = st.slider("Direction probability cutoff", 0.80, 0.99, 0.95, 0.01)
    direction_threshold = st.slider("Direction threshold (log2FC)", 0.1, 1.0, 0.5, 0.1)
    
    expected_fc_map = {
        species_1_name: species_1_fc,
        species_2_name: species_2_fc,
        species_3_name: species_3_fc
    }

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📤 Data Input", "📊 Classification Results", "📈 Metrics Dashboard", "🔬 Distribution Analysis"])

# ============================================================================
# TAB 1: DATA INPUT
# ============================================================================

with tab1:
    st.subheader("Upload Proteomics Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Flexible column mapping** - auto-detects protein_id, species, log2_fc")
        uploaded_file = st.file_uploader("Upload CSV/TSV", type=['csv', 'tsv', 'txt'])
    
    with col2:
        use_demo = st.checkbox("Use demo data", value=True)
    
    # Load data
    df = None
    if uploaded_file is not None:
        sep = '\t' if uploaded_file.name.endswith('.tsv') else ','
        try:
            df = pd.read_csv(uploaded_file, sep=sep)
            st.success(f"✅ Uploaded: {len(df)} proteins")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    if use_demo or df is None:
        np.random.seed(42)
        n_proteins = 500
        species_list = np.random.choice([species_1_name, species_2_name, species_3_name], 
                                        n_proteins, p=[0.6, 0.25, 0.15])
        demo_data = []
        for i, sp in enumerate(species_list):
            expected = expected_fc_map[sp]
            base_fc = expected + np.random.normal(0, 0.3)
            reps = base_fc + np.random.normal(0, 0.15, 5)
            demo_data.append({
                'protein_id': f'PROT_{i:04d}',
                'species': sp,
                'log2_fc': np.mean(reps),
                'replicate_values': ','.join([f'{v:.4f}' for v in reps])
            })
        df = pd.DataFrame(demo_data)
        if uploaded_file is None:
            st.info(f"📊 Demo data loaded: {len(df)} proteins")
    
    if df is not None:
        # Auto-detect columns
        protein_col, species_col, fc_col = detect_columns(df)
        
        st.divider()
        st.subheader("Column Mapping")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            protein_col = st.selectbox("Protein ID Column", df.columns.tolist(),
                index=df.columns.tolist().index(protein_col) if protein_col in df.columns else 0)
        
        with col_b:
            species_col = st.selectbox("Species Column", [None] + df.columns.tolist(),
                index=(df.columns.tolist().index(species_col) + 1) if species_col and species_col in df.columns else 0)
        
        with col_c:
            fc_col = st.selectbox("Fold-Change Column (log2)", df.columns.tolist(),
                index=df.columns.tolist().index(fc_col) if fc_col in df.columns else (1 if len(df.columns) > 1 else 0))
        
        st.divider()
        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Store mapped columns
        st.session_state['data'] = df
        st.session_state['protein_col'] = protein_col
        st.session_state['species_col'] = species_col
        st.session_state['fc_col'] = fc_col

# ============================================================================
# TAB 2: CLASSIFICATION RESULTS
# ============================================================================

with tab2:
    if 'data' not in st.session_state:
        st.warning("⚠️ Please load data in the Data Input tab first.")
    else:
        df = st.session_state['data']
        protein_col = st.session_state['protein_col']
        species_col = st.session_state['species_col']
        fc_col = st.session_state['fc_col']
        
        with st.spinner("🔬 Classifying proteins..."):
            classifications = {}
            
            for idx, row in df.iterrows():
                try:
                    # Extract values using mapped columns
                    protein_id = str(row[protein_col])
                    species = str(row[species_col]) if species_col else "Unknown"
                    observed_fc = float(row[fc_col])
                
                except (KeyError, ValueError, TypeError):
                    continue  # Skip malformed rows silently
                
                expected_fc = expected_fc_map.get(species, 0)
                
                # Generate posterior samples
                replicate_cols = [c for c in df.columns if 'replicate' in c.lower() or 'rep' in c.lower()]
                
                if replicate_cols:
                    try:
                        reps = np.array([float(row[c]) for c in replicate_cols if pd.notna(row[c])], dtype=float)
                        if len(reps) > 0:
                            posterior = np.random.normal(np.mean(reps), max(np.std(reps), 0.01) + 0.01, 1000)
                        else:
                            posterior = np.random.normal(observed_fc, 0.2, 1000)
                    except:
                        posterior = np.random.normal(observed_fc, 0.2, 1000)
                else:
                    posterior = np.random.normal(observed_fc, 0.2, 1000)
                
                tolerance = abs(expected_fc) * equiv_tolerance_pct if expected_fc != 0 else 0.5
                
                result = classify_protein(posterior, expected_fc, tolerance, 
                                          direction_threshold, direction_prob_cutoff)
                result['species'] = species
                result['protein_id'] = protein_id
                classifications[protein_id] = result
            
            results_df = pd.DataFrame(classifications).T.reset_index(drop=True)
        
        st.session_state['classifications'] = classifications
        st.session_state['results_df'] = results_df
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        category_counts = results_df['category'].value_counts()
        
        with col1:
            val = category_counts.get('CORRECT_DIRECTION_AND_MAGNITUDE', 0)
            st.metric("✅ Correct Dir & Mag", val, f"{val/len(results_df)*100:.1f}%")
        with col2:
            val = category_counts.get('CORRECT_DIRECTION_WRONG_MAGNITUDE', 0)
            st.metric("⚠️ Correct Dir Only", val, f"{val/len(results_df)*100:.1f}%")
        with col3:
            val = category_counts.get('WRONG_DIRECTION', 0)
            st.metric("❌ Wrong Direction", val, f"{val/len(results_df)*100:.1f}%")
        with col4:
            val = category_counts.get('TRUE_FALSE_POSITIVE', 0) + category_counts.get('CORRECT_UNCHANGED', 0)
            st.metric("🔵 Background", val)
        
        # Category distribution plot
        fig_cat = px.histogram(results_df, x='category', color='species', barmode='group',
                               title="Classification Distribution by Species",
                               color_discrete_sequence=px.colors.qualitative.Set2)
        fig_cat.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_cat, use_container_width=True)
        
        # Detailed results table
        st.subheader("Classification Results")
        
        filter_category = st.multiselect("Filter by category", results_df['category'].unique(), 
                                          default=results_df['category'].unique())
        filter_species = st.multiselect("Filter by species", results_df['species'].unique(),
                                         default=results_df['species'].unique())
        
        filtered_df = results_df[
            (results_df['category'].isin(filter_category)) & 
            (results_df['species'].isin(filter_species))
        ]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button("📥 Download Classifications", csv, "classifications.csv", "text/csv")

# ============================================================================
# TAB 3: METRICS DASHBOARD
# ============================================================================

with tab3:
    if 'classifications' not in st.session_state:
        st.warning("Run classification first in the Classification Results tab.")
    else:
        classifications = st.session_state['classifications']
        results_df = st.session_state['results_df']
        df = st.session_state['data']
        
        st.subheader("LFQ_bout Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        # Calculate deFDR
        deFDR = calculate_deFDR(classifications, {})
        
        with col1:
            st.metric("Directional Error FDR (deFDR)", f"{deFDR*100:.2f}%",
                     delta="PASS" if deFDR <= 0.01 else "FAIL",
                     delta_color="normal" if deFDR <= 0.01 else "inverse")
            st.caption("Target: ≤1%")
        
        # Asymmetry factor
        all_fc = results_df['mean_fc'].values.astype(float)
        asymmetry = calculate_asymmetry_factor(all_fc)
        
        with col2:
            st.metric("Asymmetry Factor", f"{asymmetry:.2f}",
                     delta="PASS" if 0.5 <= asymmetry <= 2.0 else "FAIL",
                     delta_color="normal" if 0.5 <= asymmetry <= 2.0 else "inverse")
            st.caption("Target: 0.5-2.0")
        
        # Overall RMSE
        observed_fc = results_df['mean_fc'].values.astype(float)
        expected_fc_arr = results_df['expected_fc'].values.astype(float)
        overall_rmse = np.sqrt(np.mean((observed_fc - expected_fc_arr) ** 2))
        
        with col3:
            st.metric("Overall RMSE", f"{overall_rmse:.3f}")
        
        st.divider()
        
        # Stratified RMSE
        st.subheader("Stratified RMSE by Species")
        species_labels = results_df['species'].values
        stratified = calculate_stratified_rmse(observed_fc, expected_fc_arr, species_labels)
        
        rmse_df = pd.DataFrame(stratified).T
        rmse_df.index.name = 'Species'
        st.dataframe(rmse_df.round(4), use_container_width=True)
        
        st.divider()
        
        # Process Capability Analysis
        st.subheader("Process Capability (Cpk) by Species")
        
        cpk_results = {}
        for species in results_df['species'].unique():
            mask = results_df['species'] == species
            obs = results_df.loc[mask, 'mean_fc'].values.astype(float)
            exp = expected_fc_map.get(species, 0)
            cpk_results[species] = proteomics_capability_analysis(obs, exp, equiv_tolerance_pct)
        
        cpk_df = pd.DataFrame(cpk_results).T
        cpk_df.index.name = 'Species'
        
        # Cpk visualization
        fig_cpk = go.Figure()
        for species in cpk_df.index:
            cpk_val = cpk_df.loc[species, 'Cpk']
            color = 'green' if cpk_val >= 1.33 else ('yellow' if cpk_val >= 1.0 else 'red')
            fig_cpk.add_trace(go.Bar(name=species, x=[species], y=[cpk_val], 
                                     marker_color=color, text=f"{cpk_val:.2f}", textposition='outside'))
        
        fig_cpk.add_hline(y=1.33, line_dash="dash", line_color="green", annotation_text="Industry Standard (1.33)")
        fig_cpk.add_hline(y=1.0, line_dash="dash", line_color="orange", annotation_text="Minimum (1.0)")
        fig_cpk.update_layout(title="Process Capability Index (Cpk) by Species", showlegend=False,
                              yaxis_title="Cpk", xaxis_title="Species")
        st.plotly_chart(fig_cpk, use_container_width=True)
        
        with st.expander("📋 Full Capability Analysis"):
            st.dataframe(cpk_df.round(4), use_container_width=True)

# ============================================================================
# TAB 4: DISTRIBUTION ANALYSIS
# ============================================================================

with tab4:
    if 'results_df' not in st.session_state:
        st.warning("Run classification first.")
    else:
        results_df = st.session_state['results_df']
        
        st.subheader("Fold-Change Distribution Analysis")
        
        # Observed vs Expected scatter
        fig_scatter = px.scatter(results_df, x='expected_fc', y='mean_fc', color='category',
                                 hover_data=['protein_id', 'species'],
                                 title="Observed vs Expected Fold-Changes",
                                 labels={'expected_fc': 'Expected log2FC', 'mean_fc': 'Observed log2FC'},
                                 color_discrete_sequence=px.colors.qualitative.Set1)
        
        # Add diagonal line
        min_val = min(results_df['expected_fc'].min(), results_df['mean_fc'].min()) - 0.5
        max_val = max(results_df['expected_fc'].max(), results_df['mean_fc'].max()) + 0.5
        fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                         mode='lines', name='Perfect agreement',
                                         line=dict(dash='dash', color='gray')))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Distribution by species
        st.subheader("Distribution by Species")
        fig_dist = px.violin(results_df, x='species', y='mean_fc', color='species', box=True,
                             title="Fold-Change Distribution by Species",
                             labels={'mean_fc': 'Observed log2FC'})
        
        # Add expected values as horizontal lines
        for species, exp_fc in expected_fc_map.items():
            fig_dist.add_hline(y=exp_fc, line_dash="dot", 
                              annotation_text=f"{species} expected: {exp_fc}")
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # K-S Test results
        st.subheader("Distribution Comparison Tests")
        
        ks_results = []
        for species in results_df['species'].unique():
            mask = results_df['species'] == species
            observed = results_df.loc[mask, 'mean_fc'].values.astype(float)
            expected = np.full_like(observed, expected_fc_map.get(species, 0))
            
            comparison = compare_fc_distributions(observed, expected)
            ks_results.append({
                'Species': species,
                'KS Statistic': comparison['KS']['D'],
                'KS p-value': comparison['KS']['p'],
                'n': len(observed)
            })
        
        ks_df = pd.DataFrame(ks_results)
        st.dataframe(ks_df.round(4), use_container_width=True)

# Footer
st.divider()
st.caption("Multi-Category Protein Classification Framework | Based on Triqler + TOST + LFQ_bout methodology")
