"""
Utility functions for Uncertainty Analysis Dashboard
Handles data loading, processing, and statistical computations
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime


def load_uncertainty_logs(log_path: str = "~/OWG/logs/uncertainty_logs.jsonl") -> pd.DataFrame:
    """Load and parse uncertainty logs from JSONL file"""
    log_path = os.path.expanduser(log_path)
    
    if not os.path.exists(log_path):
        return pd.DataFrame()
    
    records = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    if not records:
        return pd.DataFrame()
    
    # Flatten the nested structure
    flattened = []
    for record in records:
        base = {
            'experiment_id': record.get('experiment_id'),
            'experiment_group': record.get('experiment_group'),
            'timestamp': record.get('timestamp'),
        }
        
        # Extract model info
        model_info = record.get('model', {})
        base['grounder_model'] = model_info.get('grounder', {}).get('model_name')
        base['planner_model'] = model_info.get('planner', {}).get('model_name')
        base['ranker_model'] = model_info.get('ranker', {}).get('model_name')
        
        # Extract prompt names
        prompt_info = record.get('prompt_name', {})
        base['grounder_prompt'] = prompt_info.get('grounder')
        base['planner_prompt'] = prompt_info.get('planner')
        base['ranker_prompt'] = prompt_info.get('ranker')
        
        # Extract metadata for each stage
        metadata = record.get('metadata', {})
        for stage in ['grounder', 'planner', 'ranker']:
            stage_data = metadata.get(stage, [])
            if stage_data:
                # Aggregate metrics across steps
                confidences = [s.get('confidence', 0) for s in stage_data]
                entropies = [s.get('entropy', 0) for s in stage_data]
                
                base[f'{stage}_confidence'] = np.mean(confidences) if confidences else None
                base[f'{stage}_entropy'] = np.mean(entropies) if entropies else None
                base[f'{stage}_confidence_std'] = np.std(confidences) if len(confidences) > 1 else 0
        
        flattened.append(base)
    
    df = pd.DataFrame(flattened)
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
    
    return df


def load_batch_logs(experiments_dir: str = "~/OWG/experiments") -> pd.DataFrame:
    """Load and parse batch experiment logs"""
    experiments_dir = os.path.expanduser(experiments_dir)
    
    if not os.path.exists(experiments_dir):
        return pd.DataFrame()
    
    batch_records = []
    batch_dirs = sorted(Path(experiments_dir).glob("batch_exp_*"))
    
    for batch_dir in batch_dirs:
        batch_id = batch_dir.name
        batch_file = batch_dir / "batch_results.jsonl"
        
        if batch_file.exists():
            with open(batch_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        record['batch_id'] = batch_id
                        batch_records.append(record)
                    except json.JSONDecodeError:
                        continue
    
    if not batch_records:
        return pd.DataFrame()
    
    df = pd.DataFrame(batch_records)
    
    # Convert timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def merge_logs(uncertainty_df: pd.DataFrame, batch_df: pd.DataFrame) -> pd.DataFrame:
    """Merge uncertainty logs with batch information"""
    if uncertainty_df.empty:
        return pd.DataFrame()
    
    if batch_df.empty:
        # Add empty batch_id column if no batch data
        uncertainty_df['batch_id'] = None
        uncertainty_df['success'] = None
        uncertainty_df['attempts'] = None
        uncertainty_df['n_objects'] = None
        uncertainty_df['query'] = None
        uncertainty_df['prompt_type'] = None
        uncertainty_df['model_category'] = None
        return uncertainty_df
    
    # Merge on experiment_id
    merged = uncertainty_df.merge(
        batch_df[['experiment_id', 'batch_id', 'success', 'attempts', 
                  'n_objects', 'query', 'prompt_type', 'model_category']],
        on='experiment_id',
        how='left'
    )
    
    # Ensure success column is boolean/binary
    if 'success' in merged.columns:
        # Convert to boolean, handling various input types
        merged['success'] = merged['success'].apply(
            lambda x: bool(x) if pd.notna(x) and x != '' else np.nan
        )
        # Convert boolean to int for correlation calculations (True->1, False->0)
        merged['success_numeric'] = merged['success'].astype(float)
    
    return merged

def clean_confidence_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid confidence values (-1 indicates extraction failure)"""
    df_clean = df.copy()
    
    # List of all confidence columns
    confidence_cols = [col for col in df_clean.columns if 'confidence' in col and 'std' not in col]
    
    # Replace -1 with NaN for all confidence columns
    for col in confidence_cols:
        if col in df_clean.columns:
            df_clean.loc[df_clean[col] == -1, col] = np.nan
    
    return df_clean

def calculate_overall_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate overall confidence across all stages"""
    # First clean the data (remove -1 values)
    df = clean_confidence_data(df)
    
    confidence_cols = [col for col in df.columns if '_confidence' in col and '_std' not in col]
    
    if confidence_cols:
        df['overall_confidence'] = df[confidence_cols].mean(axis=1)
    else:
        df['overall_confidence'] = None
    
    return df

def filter_dataframe(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to dataframe"""
    filtered = df.copy()
    
    if filters.get('batch_ids') and None not in filters['batch_ids']:
        filtered = filtered[filtered['batch_id'].isin(filters['batch_ids'])]
    
    if filters.get('experiment_groups'):
        filtered = filtered[filtered['experiment_group'].isin(filters['experiment_groups'])]
    
    if filters.get('prompt_types'):
        filtered = filtered[filtered['prompt_type'].isin(filters['prompt_types'])]
    
    if filters.get('model_names'):
        # Check any of the model columns
        model_mask = (
            filtered['grounder_model'].isin(filters['model_names']) |
            filtered['planner_model'].isin(filters['model_names']) |
            filtered['ranker_model'].isin(filters['model_names'])
        )
        filtered = filtered[model_mask]
    
    if filters.get('model_categories'):
        filtered = filtered[filtered['model_category'].isin(filters['model_categories'])]
    
    if filters.get('success_filter') != 'All':
        if filters.get('success_filter') == 'Success':
            filtered = filtered[filtered['success'] == True]
        elif filters.get('success_filter') == 'Failure':
            filtered = filtered[filtered['success'] == False]
    
    if filters.get('date_range'):
        start_date, end_date = filters['date_range']
        if 'date' in filtered.columns:
            filtered = filtered[
                (filtered['date'] >= start_date) & 
                (filtered['date'] <= end_date)
            ]
    
    if filters.get('n_objects_range'):
        min_obj, max_obj = filters['n_objects_range']
        if 'n_objects' in filtered.columns:
            filtered = filtered[
                (filtered['n_objects'] >= min_obj) & 
                (filtered['n_objects'] <= max_obj)
            ]
    
    if filters.get('query_search'):
        if 'query' in filtered.columns:
            filtered = filtered[
                filtered['query'].str.contains(filters['query_search'], 
                                               case=False, na=False)
            ]
    
    return filtered


def calculate_success_rate(df: pd.DataFrame, group_by: Optional[str] = None) -> pd.DataFrame:
    """Calculate success rate, optionally grouped by a column"""
    if df.empty or 'success' not in df.columns:
        return pd.DataFrame()
    
    if group_by and group_by in df.columns:
        result = df.groupby(group_by).agg({
            'success': ['sum', 'count', 'mean']
        }).reset_index()
        result.columns = [group_by, 'successes', 'total', 'success_rate']
    else:
        result = pd.DataFrame([{
            'successes': df['success'].sum(),
            'total': len(df),
            'success_rate': df['success'].mean()
        }])
    
    return result


def perform_ttest(group1: pd.Series, group2: pd.Series) -> Tuple[float, float, str]:
    """Perform independent t-test between two groups"""
    # Remove NaN values
    g1 = group1.dropna()
    g2 = group2.dropna()
    
    if len(g1) < 2 or len(g2) < 2:
        return np.nan, np.nan, "Insufficient data"
    
    # Check for normality (Shapiro-Wilk test)
    _, p1 = stats.shapiro(g1) if len(g1) < 5000 else (None, 0.05)
    _, p2 = stats.shapiro(g2) if len(g2) < 5000 else (None, 0.05)
    
    # If data is not normal, use Mann-Whitney U test
    if p1 < 0.05 or p2 < 0.05:
        statistic, p_value = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        test_type = "Mann-Whitney U"
    else:
        statistic, p_value = stats.ttest_ind(g1, g2)
        test_type = "Independent t-test"
    
    return statistic, p_value, test_type


def perform_anova(groups: List[pd.Series], group_names: List[str]) -> Tuple[float, float, pd.DataFrame]:
    """Perform one-way ANOVA and post-hoc tests"""
    # Remove NaN and filter groups with sufficient data
    cleaned_groups = [g.dropna() for g in groups]
    valid_indices = [i for i, g in enumerate(cleaned_groups) if len(g) >= 2]
    
    if len(valid_indices) < 2:
        return np.nan, np.nan, pd.DataFrame()
    
    cleaned_groups = [cleaned_groups[i] for i in valid_indices]
    valid_names = [group_names[i] for i in valid_indices]
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*cleaned_groups)
    
    # Post-hoc pairwise comparisons (Tukey's HSD approximation using t-tests with Bonferroni correction)
    n_comparisons = len(cleaned_groups) * (len(cleaned_groups) - 1) // 2
    alpha_corrected = 0.05 / n_comparisons if n_comparisons > 0 else 0.05
    
    posthoc_results = []
    for i in range(len(cleaned_groups)):
        for j in range(i + 1, len(cleaned_groups)):
            stat, p = stats.ttest_ind(cleaned_groups[i], cleaned_groups[j])
            posthoc_results.append({
                'group1': valid_names[i],
                'group2': valid_names[j],
                'statistic': stat,
                'p_value': p,
                'significant': p < alpha_corrected
            })
    
    posthoc_df = pd.DataFrame(posthoc_results)
    
    return f_stat, p_value, posthoc_df


def calculate_calibration_metrics(df: pd.DataFrame, confidence_col: str = 'overall_confidence') -> Dict:
    """Calculate calibration metrics (ECE, MCE, Brier score)"""
    if df.empty or confidence_col not in df.columns or 'success' not in df.columns:
        return {}
    
    # Remove rows with missing values
    valid_data = df[[confidence_col, 'success']].dropna()
    
    if len(valid_data) < 10:
        return {}
    
    confidences = valid_data[confidence_col].values
    outcomes = valid_data['success'].astype(int).values
    
    # Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    bin_data = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = outcomes[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
            
            bin_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'confidence': avg_confidence_in_bin,
                'accuracy': accuracy_in_bin,
                'count': in_bin.sum()
            })
    
    # Brier Score
    brier_score = np.mean((confidences - outcomes) ** 2)
    
    return {
        'ece': ece,
        'mce': mce,
        'brier_score': brier_score,
        'bin_data': bin_data
    }


def get_significance_marker(p_value: float) -> str:
    """Return significance marker based on p-value"""
    if pd.isna(p_value):
        return ""
    elif p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def calculate_correlation(df: pd.DataFrame, col1: str, col2: str) -> Tuple[float, float]:
    """Calculate Pearson correlation between two columns"""
    if df.empty or col1 not in df.columns or col2 not in df.columns:
        return np.nan, np.nan
    
    valid_data = df[[col1, col2]].dropna()
    
    if len(valid_data) < 3:
        return np.nan, np.nan
    
    # Convert to numeric and ensure proper dtype
    try:
        col1_numeric = pd.to_numeric(valid_data[col1], errors='coerce')
        col2_numeric = pd.to_numeric(valid_data[col2], errors='coerce')
        
        # Remove any remaining NaN after conversion
        mask = ~(col1_numeric.isna() | col2_numeric.isna())
        col1_numeric = col1_numeric[mask]
        col2_numeric = col2_numeric[mask]
        
        if len(col1_numeric) < 3:
            return np.nan, np.nan
        
        corr, p_value = stats.pearsonr(col1_numeric, col2_numeric)
        
        return corr, p_value
    except Exception:
        return np.nan, np.nan
