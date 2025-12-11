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

def safe_column_exists(df: pd.DataFrame, col: str) -> bool:
    """Safely check if column exists and has non-null values"""
    return col in df.columns and not df[col].isna().all()

def load_experiment_metrics(metrics_path: str = "~/OWG/logs/experiment_metrics.jsonl") -> pd.DataFrame:
    """Load and parse experiment metrics (success, attempts, etc.)"""
    metrics_path = os.path.expanduser(metrics_path)
    
    if not os.path.exists(metrics_path):
        return pd.DataFrame()
    
    records = []
    with open(metrics_path, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # Convert timestamp if exists
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            pass
    
    # Ensure success is boolean
    if 'success' in df.columns:
        df['success'] = df['success'].apply(
            lambda x: bool(x) if pd.notna(x) and x != '' else np.nan
        )
    
    return df

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
            'n_objects': record.get('n_objects'),  # ADD THIS - extract from uncertainty log
        }

        # Extract model_category - handle both list and string
        model_category = record.get('model_category')
        if isinstance(model_category, list):
            # Store as comma-separated string for filtering, and keep original list
            base['model_category'] = ','.join(model_category) if model_category else None
            base['model_category_list'] = model_category
        else:
            base['model_category'] = model_category
            base['model_category_list'] = [model_category] if model_category else []
        
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
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
        except Exception:
            pass
    
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
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            pass
    
    return df


def merge_logs(uncertainty_df: pd.DataFrame, batch_df: pd.DataFrame, metrics_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Merge uncertainty logs with batch information and experiment metrics.
    
    Note: This handles one-to-many relationships:
    - One experiment_id can have multiple uncertainty log entries (multiple queries)
    - One experiment_id can have multiple metric entries (multiple actions)
    
    We aggregate metrics per experiment_id before merging.
    """
    if uncertainty_df.empty:
        return pd.DataFrame()
    
    # Extract n_objects and model_category from uncertainty logs if they exist
    if 'n_objects' not in uncertainty_df.columns:
        uncertainty_df['n_objects'] = None
    if 'model_category' not in uncertainty_df.columns:
        uncertainty_df['model_category'] = None
    
    # Start with uncertainty_df as base (keep all records, including multiple queries per experiment)
    merged = uncertainty_df.copy()
    
    # Add a unique row identifier for each uncertainty log entry
    merged['log_entry_id'] = range(len(merged))
    
    # Merge batch data if available
    if not batch_df.empty:
        # Batch data is typically one-to-one with experiment_id
        batch_df_dedup = batch_df.drop_duplicates(subset=['experiment_id'], keep='last')
        
        batch_cols = ['experiment_id', 'batch_id', 'query', 'prompt_type']
        
        if 'n_objects' in batch_df_dedup.columns:
            batch_cols.append('n_objects')
        if 'model_category' in batch_df_dedup.columns:
            batch_cols.append('model_category')
        
        batch_cols_existing = [col for col in batch_cols if col in batch_df_dedup.columns]
        
        merged = merged.merge(
            batch_df_dedup[batch_cols_existing],
            on='experiment_id',
            how='left',
            suffixes=('_uncert', '_batch')
        )
        
        # Resolve conflicts
        if 'n_objects_batch' in merged.columns:
            merged['n_objects'] = merged['n_objects_batch'].fillna(merged['n_objects_uncert'])
            merged.drop(['n_objects_uncert', 'n_objects_batch'], axis=1, inplace=True)
        
        if 'model_category_batch' in merged.columns:
            merged['model_category'] = merged['model_category_batch'].fillna(merged['model_category_uncert'])
            merged.drop(['model_category_uncert', 'model_category_batch'], axis=1, inplace=True)
    else:
        merged['batch_id'] = None
        merged['query'] = None
        merged['prompt_type'] = None
    
    # Merge experiment metrics - AGGREGATE FIRST
    if metrics_df is not None and not metrics_df.empty:
        # Aggregate metrics per experiment_id
        # Success = True if ANY action succeeded (or custom logic)
        # Attempts = total number of actions/attempts
        
        metrics_agg = metrics_df.groupby('experiment_id').agg({
            'success': lambda x: x.any() if x.notna().any() else np.nan,  # True if any action succeeded
            'attempts': 'sum' if 'attempts' in metrics_df.columns else 'count'  # Total attempts
        }).reset_index()
        
        # Alternative: Take the LAST action's success status (final outcome)
        # metrics_agg = metrics_df.sort_values('timestamp').groupby('experiment_id').last().reset_index()
        
        merged = merged.merge(
            metrics_agg,
            on='experiment_id',
            how='left'
        )
        
        # Ensure success column is boolean/binary
        if 'success' in merged.columns:
            merged['success'] = merged['success'].apply(
                lambda x: bool(x) if pd.notna(x) and x != '' else np.nan
            )
            merged['success_numeric'] = merged['success'].astype(float)
    else:
        merged['success'] = None
        merged['attempts'] = None
    
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
    
    # Batch ID filter - include None/NaN if 'All' selected
    if filters.get('batch_ids') and None not in filters['batch_ids']:
        if 'batch_id' in filtered.columns:
            filtered = filtered[filtered['batch_id'].isin(filters['batch_ids'])]
    
    # Experiment group filter - only filter if groups are selected AND column exists
    if filters.get('experiment_groups'):
        if 'experiment_group' in filtered.columns:
            # Include rows where experiment_group matches OR is NaN (individual logs)
            mask = filtered['experiment_group'].isin(filters['experiment_groups']) | filtered['experiment_group'].isna()
            filtered = filtered[mask]
    
    # Prompt type filter - include NaN values
    if filters.get('prompt_types'):
        if 'prompt_type' in filtered.columns:
            mask = filtered['prompt_type'].isin(filters['prompt_types']) | filtered['prompt_type'].isna()
            filtered = filtered[mask]
    
    # Model name filter - include NaN values
    if filters.get('model_names'):
        model_mask = pd.Series([False] * len(filtered), index=filtered.index)
        has_any_model = False
        for col in ['grounder_model', 'planner_model', 'ranker_model']:
            if col in filtered.columns:
                has_any_model = True
                # Match selected models OR is NaN
                model_mask |= filtered[col].isin(filters['model_names']) | filtered[col].isna()
        
        if has_any_model:
            filtered = filtered[model_mask]
    
    # Model category filter - handle both string and list (comma-separated)
    if filters.get('model_categories'):
        if 'model_category' in filtered.columns:
            selected_cats = filters['model_categories']
            
            # Create mask that checks if any selected category is in the model_category
            def matches_category(cat_value):
                if pd.isna(cat_value):
                    return True  # Include NaN
                # Split comma-separated categories
                cats = [c.strip() for c in str(cat_value).split(',')]
                # Check if any selected category matches
                return any(sel_cat in cats for sel_cat in selected_cats)
            
            mask = filtered['model_category'].apply(matches_category)
            filtered = filtered[mask]
    
    # Success filter - only apply if not 'All'
    if filters.get('success_filter') != 'All':
        if 'success' in filtered.columns:
            if filters.get('success_filter') == 'Success':
                filtered = filtered[filtered['success'] == True]
            elif filters.get('success_filter') == 'Failure':
                filtered = filtered[filtered['success'] == False]
            # Note: NaN values will be excluded when filtering by Success/Failure
    
    # Date range filter
    if filters.get('date_range'):
        start_date, end_date = filters['date_range']
        if 'date' in filtered.columns:
            # Include rows within date range OR with NaN dates
            mask = ((filtered['date'] >= start_date) & (filtered['date'] <= end_date)) | filtered['date'].isna()
            filtered = filtered[mask]
    
    # Number of objects filter
    if filters.get('n_objects_range'):
        min_obj, max_obj = filters['n_objects_range']
        if 'n_objects' in filtered.columns:
            # Include rows within range OR with NaN n_objects
            mask = ((filtered['n_objects'] >= min_obj) & (filtered['n_objects'] <= max_obj)) | filtered['n_objects'].isna()
            filtered = filtered[mask]
    
    # Query search filter
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