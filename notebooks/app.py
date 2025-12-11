# app.py
import sys, os
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from owg_mod.prompt_library import SystemPromptLibrary
from owg_mod.model_utils_litellm import check_litellm
from owg_mod.app_utils import (
    safe_column_exists,
    load_experiment_metrics,  # ADD THIS
    load_uncertainty_logs,
    load_batch_logs,
    merge_logs,
    calculate_overall_confidence,
    filter_dataframe,
    calculate_success_rate,
    perform_ttest,
    perform_anova,
    calculate_calibration_metrics,
    get_significance_marker,
    calculate_correlation
)

# Paths for metrics
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_PATH = os.path.join(BASE_DIR, "logs", "litellm_logs.jsonl")
METRICS_PATH = os.path.join(BASE_DIR, "logs", "experiment_metrics.jsonl")
LOG_PATH_UNCERT = os.path.join(BASE_DIR, "logs", "uncertainty_logs.jsonl")
BATCH_EXPERIMENTS_DIR = "/home/owner/OWG/experiments"

# Paths for prompts
UNCERTAINTY_DIR = os.path.join(BASE_DIR, "prompts/uncertainty_aware")
USER_PROMPT_DIR = os.path.join(BASE_DIR, "prompts/user_defined")

# Path to user defined pybullet config
CONFIG_DIR = os.path.join(BASE_DIR, "config/pyb/user_defined")

# Ensure directories exist
os.makedirs(USER_PROMPT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

default_prompt_lib = SystemPromptLibrary(UNCERTAINTY_DIR)
custom_prompt_lib = SystemPromptLibrary(USER_PROMPT_DIR)
default_available_prompts = default_prompt_lib.list_available_prompts()
custom_available_prompts = custom_prompt_lib.list_available_prompts()

st.set_page_config(page_title="OWG Experiment Dashboard", layout="wide")
st.title("🧠 Open World Grasping — LLM Experiment Dashboard")

#Debugging
st.write("Resolved paths: (Debugging)")
st.code(f"LOG_PATH = {LOG_PATH}\nMETRICS_PATH = {METRICS_PATH}\nUNCERTAINTY_PATH = {LOG_PATH_UNCERT}\nPROMPTS_PATH = {UNCERTAINTY_DIR} & {USER_PROMPT_DIR}")

tabs = st.tabs(["🔍 Experiment Logs", "🧩 Prompt Engineering", "📈 Metrics Overview", "🧠 Uncertainty Analysis", "RUN EXPERIMENT"])

# --- TAB 1: LITELLM LOGS ---
with tabs[0]:
    st.subheader("Recent LiteLLM Calls")
    auto_refresh = st.toggle("🔄 Auto-refresh every 15s", value=True)

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            logs = [json.loads(line) for line in f if line.strip()]

        if len(logs) == 0:
            st.info("Log file found but empty.")
        else:
            # --- Flatten JSON entries to handle nested metadata cleanly ---
            df = pd.json_normalize(logs, sep=".")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp", ascending=False)

            # --- Local Filters (tab-only, replaces sidebar) ---
            with st.expander("🔍 Filter LiteLLM Logs", expanded=True):
                models = sorted(df["model"].dropna().unique()) if "model" in df.columns else []
                statuses = sorted(df["status"].dropna().unique()) if "status" in df.columns else []
                users = sorted([u for u in df["user"].dropna().unique() if u])  # skip empty strings

                col1, col2, col3 = st.columns(3)
                selected_model = col1.multiselect("Model", models, default=models)
                selected_status = col2.multiselect("Status", statuses, default=statuses)
                selected_user = col3.multiselect("User", ["(none)"] + users, default=["(none)"] + users)

            df["user"] = df["user"].fillna("(none)")
            filtered_df = df[
                (df["model"].isin(selected_model)) &
                (df["status"].isin(selected_status)) &
                (df["user"].isin(selected_user))
            ]

            # --- Display ---
            st.markdown("### 📋 Filtered LiteLLM Logs")

            default_cols = ["timestamp", "status", "model", "cost", "response"]
            optional_cols = sorted([c for c in filtered_df.columns if c not in default_cols])

            with st.expander("⚙️ Show advanced columns"):
                extra_cols = st.multiselect(
                    "Select additional fields to view",
                    options=optional_cols,
                    default=[]
                )

            display_cols = [c for c in default_cols + extra_cols if c in filtered_df.columns]

            st.dataframe(filtered_df[display_cols], width='stretch', height=400)

            # --- Summary Stats ---
            st.markdown("### 📊 Summary Stats")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Calls", len(filtered_df))
            if "status" in filtered_df.columns:
                col2.metric("Success Rate", f"{(filtered_df['status'] == 'success').mean() * 100:.1f}%")
            else:
                col2.metric("Success Rate", "N/A")
            col3.metric(
                "Total Cost ($)",
                f"{filtered_df['cost'].sum():.6f}" if "cost" in filtered_df.columns else "N/A",
            )

            # --- Charts ---
            st.markdown("### 📈 Cost & Activity Trends")

            if "timestamp" in filtered_df.columns:
                cost_group = (
                    filtered_df.groupby(pd.Grouper(key="timestamp", freq="5min"))
                    .agg({"cost": "sum"})
                    .reset_index()
                )
                cost_chart = (
                    alt.Chart(cost_group)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("timestamp:T", title="Time"),
                        y=alt.Y("cost:Q", title="Total Cost ($)"),
                        tooltip=["timestamp:T", "cost:Q"]
                    )
                    .properties(height=300)
                )

                count_group = (
                    filtered_df.groupby([pd.Grouper(key="timestamp", freq="5min"), "status"])
                    .size()
                    .reset_index(name="count")
                )
                count_chart = (
                    alt.Chart(count_group)
                    .mark_bar()
                    .encode(
                        x=alt.X("timestamp:T", title="Time"),
                        y=alt.Y("count:Q", title="Requests"),
                        color="status:N",
                        tooltip=["timestamp:T", "status:N", "count:Q"]
                    )
                    .properties(height=300)
                )

                st.altair_chart(cost_chart, width='stretch')
                st.altair_chart(count_chart, width='stretch')

            st.info("✅ Logs loaded successfully. Use filter expander above for model, status, or user.")
    else:
        st.warning("No log file found. Ensure LiteLLM writes to `logs/litellm_logs.jsonl`.")


# --- TAB 2: PROMPT ENGINEERING (placeholder) ---
with tabs[1]:
    # --- Run Experiment ---
    st.subheader("🚀 Run Experiment with This Prompt")
    st.write("Design, modify, and experiment with OWG prompts interactively.")
    
    # --- Create two columns ---
    left_col, right_col = st.columns([1, 1])
    
    # ==================== LEFT SIDE: PROMPT PREVIEW ====================
    with left_col:
        st.markdown("### 📚 Prompt Library")
        
        # --- Base Templates Section ---
        st.markdown("#### 📂 Base Templates")
        system_prompts = [f.replace(".txt", "") for f in os.listdir(default_prompt_lib.prompt_dir) if f.endswith(".txt")]
        
        if not system_prompts:
            st.info("No base templates found.")
        else:
            for prompt_name in sorted(system_prompts):
                with st.expander(f"📄 {prompt_name}"):
                    filepath = os.path.join(default_prompt_lib.prompt_dir, f"{prompt_name}.txt")
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                        st.code(content, language="text")
                        # if st.button(f"📋 Copy to clipboard", key=f"copy_system_{prompt_name}"):
                            # st.code(content, language="text")
                            # st.success(f"✅ Select and copy the text above!")
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
        
        st.markdown("---")
        
        # --- User Defined Section ---
        st.markdown("#### 👤 User Defined Prompts")
        user_prompts = [f.replace(".txt", "") for f in os.listdir(USER_PROMPT_DIR) if f.endswith(".txt")] if os.path.exists(USER_PROMPT_DIR) else []
        
        if not user_prompts:
            st.info("No user-defined prompts yet. Create one on the right! →")
        else:
            for prompt_name in sorted(user_prompts):
                with st.expander(f"📄 {prompt_name}"):
                    filepath = os.path.join(USER_PROMPT_DIR, f"{prompt_name}.txt")
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                        st.code(content, language="text")
                        if st.button(f"🗑️ Delete", key=f"delete_user_{prompt_name}"):
                            os.remove(filepath)
                            st.success(f"Deleted {prompt_name}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
    
    # ==================== RIGHT SIDE: CREATE & SAVE ====================
    with right_col:
        st.markdown("### ✍️ Create New Prompt")
        
        # --- Editable Text Area ---
        st.markdown("#### 📝 Prompt Content")
        user_prompt_text = st.text_area(
            "Write or edit your prompt:",
            value="",
            height=350,
            key="prompt_creator"
        )
        
        # --- Save Section ---
        st.markdown("#### 💾 Save Prompt")
        save_col1, save_col2 = st.columns([3, 1])
        with save_col1:
            user_filename = st.text_input("File name (without .txt):", "")
        with save_col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            save_btn = st.button("💾 Save", use_container_width=True)
        
        if save_btn:
            if not user_filename.strip():
                st.error("⚠️ Please enter a file name.")
            elif not user_prompt_text.strip():
                st.error("⚠️ Prompt content is empty.")
            else:
                os.makedirs(USER_PROMPT_DIR, exist_ok=True)
                filepath = os.path.join(USER_PROMPT_DIR, f"{user_filename}.txt")
                with open(filepath, "w") as f:
                    f.write(user_prompt_text)
                st.success(f"✅ Prompt saved as `{user_filename}.txt`")
                st.rerun()

# --- TAB 3: METRICS OVERVIEW (placeholder) ---
with tabs[2]:
    st.subheader("📈 Experiment Metrics Dashboard")

    if not os.path.exists(METRICS_PATH):
        st.warning("No experiment log found. Run experiments to populate tracker logs.")
    else:
        with open(METRICS_PATH, "r") as f:
            logs = [json.loads(line) for line in f if line.strip()]

        if logs:
            df_grasp = pd.DataFrame(logs)
            df_grasp["timestamp"] = pd.to_datetime(df_grasp["timestamp"], errors="coerce")
            df_grasp = df_grasp.sort_values("timestamp", ascending=False)

            # --- Display Table ---
            st.markdown("### 📋 Recent Grasp Attempts")
            st.dataframe(
                df_grasp[["timestamp", "object_id", "position", "success", "grasp_type", "retries"]],
                width="stretch",
                height=400
            )

            # --- Compute Summary ---
            total = len(df_grasp)
            success_rate = df_grasp["success"].mean() * 100 if total > 0 else 0
            retries_avg = df_grasp["retries"].mean() if total > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Grasps", total)
            col2.metric("Success Rate", f"{success_rate:.1f}%")
            col3.metric("Avg Retries", f"{retries_avg:.2f}")

            # --- Per Object Chart ---
            success_per_object = (
                df_grasp.groupby("object_id")["success"]
                .mean()
                .reset_index()
                .rename(columns={"success": "success_rate"})
            )
            if not success_per_object.empty:
                st.markdown("### 🧱 Success Rate per Object")
                chart = (
                    alt.Chart(success_per_object)
                    .mark_bar()
                    .encode(
                        x=alt.X("object_id:N", title="Object ID"),
                        y=alt.Y("success_rate:Q", title="Success Rate", axis=alt.Axis(format="%")),
                        tooltip=["object_id:N", alt.Tooltip("success_rate:Q", format=".2%")],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, width='stretch')

            # --- Timeline Chart ---
            st.markdown("### ⏱️ Grasp Attempts Over Time")
            timeline = (
                alt.Chart(df_grasp)
                .mark_circle(size=80)
                .encode(
                    x=alt.X("timestamp:T", title="Time"),
                    y=alt.Y("success:N", title="Success (True/False)"),
                    color="success:N",
                    tooltip=["timestamp:T", "object_id:N", "success:N", "retries:Q"],
                )
                .properties(height=300)
            )
            st.altair_chart(timeline, width='stretch')

        else:
            st.info("Empty grasp logs.")

# 📊 Refined Uncertainty Analysis Dashboard with Statistical Tests
with tabs[3]:
    st.title("🎯 Uncertainty Analysis Dashboard")
    st.markdown("*Open World Grasping (OWG) Project - Uncertainty Estimation Module*")
    
    # Load data
    with st.spinner("Loading data..."):
        uncertainty_df = load_uncertainty_logs()
        batch_df = load_batch_logs()
        metrics_df = load_experiment_metrics()  # ADD THIS LINE
        
        if uncertainty_df.empty:
            st.error("❌ No uncertainty logs found at ~/OWG/logs/uncertainty_logs.jsonl")
            st.info("Please ensure experiments have been run and logged.")

        # DEBUG: Show raw counts
        with st.expander("🔧 Debug Info - Raw Data Counts", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Uncertainty Logs", len(uncertainty_df))
                if 'experiment_id' in uncertainty_df.columns:
                    unique_ids = uncertainty_df['experiment_id'].nunique()
                    st.caption(f"Unique experiment IDs: {unique_ids}")
                    st.caption(f"Avg queries per exp: {len(uncertainty_df)/unique_ids:.1f}")
            with col2:
                st.metric("Batch Logs", len(batch_df))
                if not batch_df.empty and 'experiment_id' in batch_df.columns:
                    st.caption(f"Unique IDs: {batch_df['experiment_id'].nunique()}")
            with col3:
                st.metric("Metrics Logs", len(metrics_df))
                if not metrics_df.empty and 'experiment_id' in metrics_df.columns:
                    unique_ids = metrics_df['experiment_id'].nunique()
                    st.caption(f"Unique experiment IDs: {unique_ids}")
                    st.caption(f"Avg actions per exp: {len(metrics_df)/unique_ids:.1f}")
            
            st.info("""
            📝 Note: Multiple entries per experiment ID are expected:
            - **Uncertainty logs**: Multiple queries per experiment (e.g., 3 queries in batch tests)
            - **Metrics logs**: Multiple actions per experiment (e.g., remove then pick)
            
            The dashboard aggregates metrics per experiment while preserving all uncertainty query records.
            """)
        
        # Merge datasets
        df = merge_logs(uncertainty_df, batch_df, metrics_df)
        df = calculate_overall_confidence(df)
        
        # Validate essential columns exist
        required_cols = ['experiment_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info("The log file may be corrupted or in an unexpected format.")
        
        if len(df) == 0:
            st.warning("⚠️ No valid experiments found in the logs.")
            st.info("The log file exists but contains no valid data.")
    
    # ===== FILTER PANEL =====
    with st.expander("🔍 **Advanced Filters**", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Batch ID filter
            if 'batch_id' in df.columns:
                batch_ids = ['All'] + sorted([b for b in df['batch_id'].dropna().unique()])
            else:
                batch_ids = ['All']
            
            selected_batches = st.multiselect(
                "Batch ID",
                options=batch_ids,
                default=['All']
            )
            
            # Experiment Group filter
            if 'experiment_group' in df.columns:
                exp_groups = df['experiment_group'].dropna().unique().tolist()
            else:
                exp_groups = []
            
            selected_groups = st.multiselect(
                "Experiment Group",
                options=exp_groups,
                default=exp_groups
            )
        
        with col2:
            # Prompt Type filter
            if 'prompt_type' in df.columns:
                prompt_types = df['prompt_type'].dropna().unique().tolist()
            else:
                prompt_types = []
            
            selected_prompts = st.multiselect(
                "Prompt Type",
                options=prompt_types,
                default=prompt_types
            )
            
            # Model Name filter
            all_models = set()
            for col in ['grounder_model', 'planner_model', 'ranker_model']:
                if col in df.columns:
                    all_models.update(df[col].dropna().unique())
            
            selected_models = st.multiselect(
                "Model Name",
                options=sorted(all_models) if all_models else [],
                default=sorted(all_models) if all_models else []
            )
        
        with col3:
            # Model Category filter - extract all unique categories from comma-separated values
            if 'model_category' in df.columns:
                all_categories = set()
                for cat_value in df['model_category'].dropna().unique():
                    if cat_value:  # Skip None/empty
                        # Split comma-separated categories
                        cats = [c.strip() for c in str(cat_value).split(',') if c.strip()]
                        all_categories.update(cats)
                model_cats = sorted(all_categories)
            else:
                model_cats = []
            
            selected_categories = st.multiselect(
                "Model Category",
                options=model_cats,
                default=model_cats,
                help="Shows all categories used across all pipeline stages"
            )
            
            # Success filter
            success_filter = st.selectbox(
                "Success Status",
                options=['All', 'Success', 'Failure'],
                index=0
            )
        
        with col4:
            # Date range filter
            if safe_column_exists(df, 'date'):
                min_date = df['date'].min()
                max_date = df['date'].max()
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            else:
                date_range = None
                st.info("Date filter unavailable")
            
            # Number of objects range
            if safe_column_exists(df, 'n_objects'):
                min_obj = int(df['n_objects'].min())
                max_obj = int(df['n_objects'].max())
                
                # Only show slider if there's a range
                if min_obj < max_obj:
                    n_objects_range = st.slider(
                        "Number of Objects",
                        min_value=min_obj,
                        max_value=max_obj,
                        value=(min_obj, max_obj)
                    )
                else:
                    # If all values are the same, just display it
                    st.info(f"Number of Objects: {min_obj}")
                    n_objects_range = (min_obj, max_obj)
            else:
                n_objects_range = None
        
        # Query search
        query_search = st.text_input("🔎 Search Query Text", "")
    
    # Apply filters
    filters = {
        'batch_ids': None if 'All' in selected_batches else selected_batches,
        'experiment_groups': selected_groups if selected_groups else None,
        'prompt_types': selected_prompts if selected_prompts else None,
        'model_names': selected_models if selected_models else None,
        'model_categories': selected_categories if selected_categories else None,
        'success_filter': success_filter,
        'date_range': date_range if date_range and len(date_range) == 2 else None,
        'n_objects_range': n_objects_range,
        'query_search': query_search if query_search else None
    }
    
    filtered_df = filter_dataframe(df, filters)
    
    if filtered_df.empty:
        st.warning("⚠️ No data matches the selected filters.")
    
    st.success(f"✅ **{len(filtered_df)}** experiments loaded (filtered from {len(df)} total)")

    # Show breakdown of batch vs individual experiments
    if 'batch_id' in filtered_df.columns:
        n_batch = filtered_df['batch_id'].notna().sum()
        n_individual = filtered_df['batch_id'].isna().sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📦 **{n_batch}** batch experiments")
        with col2:
            st.info(f"📝 **{n_individual}** individual experiments")
    
    # ===== SECTION A: KEY METRICS OVERVIEW =====
    st.markdown("---")
    st.header("📊 Key Metrics Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_experiments = len(filtered_df)
        st.metric("Total Experiments", f"{total_experiments:,}")
    
    with col2:
        if safe_column_exists(filtered_df, 'success'):
            success_rate = filtered_df['success'].mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("Success Rate", "N/A")
    
    with col3:
        if safe_column_exists(filtered_df, 'overall_confidence'):
            avg_confidence = filtered_df['overall_confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        else:
            st.metric("Avg Confidence", "N/A")
    
    with col4:
        if safe_column_exists(filtered_df, 'attempts'):
            avg_attempts = filtered_df['attempts'].mean()
            st.metric("Avg Attempts", f"{avg_attempts:.2f}")
        else:
            st.metric("Avg Attempts", "N/A")
    
    with col5:
        if 'batch_id' in filtered_df.columns:
            unique_batches = filtered_df['batch_id'].dropna().unique()
            st.metric("Unique Batches", len(unique_batches))
        else:
            st.metric("Unique Batches", "0")
    
    # ===== SECTION B: PRIORITY ANALYSIS =====
    st.markdown("---")
    st.header("🎯 Priority Analysis")
    
    # B1: Success Rate vs Confidence Correlation
    st.subheader("1️⃣ Success Rate vs Confidence Correlation")
    
    if safe_column_exists(filtered_df, 'success') and safe_column_exists(filtered_df, 'overall_confidence'):
        # Filter out invalid confidence values and ensure clean data
        plot_df = filtered_df[
            (filtered_df['overall_confidence'].notna()) & 
            (filtered_df['overall_confidence'] >= 0) &  # Remove -1 and negative values
            (filtered_df['overall_confidence'] <= 1) &  # Ensure within valid range
            (filtered_df['success'].notna())
        ].copy()
        
        # Convert success to numeric (0 or 1)
        plot_df['success_numeric'] = plot_df['success'].astype(int)
        
        if len(plot_df) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Scatter plot
                fig = px.scatter(
                    plot_df,
                    x='overall_confidence',
                    y='success_numeric',
                    color='experiment_group' if safe_column_exists(plot_df, 'experiment_group') else None,
                    hover_data=['experiment_id', 'prompt_type', 'model_category'] if all(safe_column_exists(plot_df, c) for c in ['experiment_id', 'prompt_type', 'model_category']) else ['experiment_id'],
                    title="Success vs Overall Confidence",
                    labels={'overall_confidence': 'Overall Confidence', 'success_numeric': 'Success (0=Fail, 1=Success)'},
                    trendline="ols"
                )
                fig.update_layout(height=400)
                fig.update_yaxes(range=[-0.1, 1.1])  # Fix y-axis range
                fig.update_xaxes(range=[0, 1])  # Fix x-axis range
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Correlation statistics
                corr, p_value = calculate_correlation(plot_df, 'overall_confidence', 'success_numeric')
                
                st.markdown("**Correlation Statistics:**")
                st.metric("Pearson r", f"{corr:.3f}" if not np.isnan(corr) else "N/A")
                st.metric("p-value", f"{p_value:.4f}" if not np.isnan(p_value) else "N/A")
                st.metric("Significance", get_significance_marker(p_value))
                
                if not np.isnan(corr):
                    if abs(corr) < 0.3:
                        strength = "weak"
                    elif abs(corr) < 0.7:
                        strength = "moderate"
                    else:
                        strength = "strong"
                    st.info(f"Correlation is **{strength}** and **{get_significance_marker(p_value) if p_value < 0.05 else 'not significant'}**")
        else:
            st.warning("⚠️ No valid data available after filtering invalid confidence values")
    else:
        st.info("Success or confidence data not available")
    
    # B2: Performance by Model Type
    st.subheader("2️⃣ Performance by Model Type")
    
    if safe_column_exists(filtered_df, 'model_category'):
        model_perf = calculate_success_rate(filtered_df, 'model_category')
        
        if not model_perf.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Success rate by model category
                fig = px.bar(
                    model_perf,
                    x='model_category',
                    y='success_rate',
                    text='success_rate',
                    title="Success Rate by Model Category",
                    labels={'model_category': 'Model Category', 'success_rate': 'Success Rate'}
                )
                fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                fig.update_layout(height=400, yaxis_range=[0, 1.1])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence by model category
                if safe_column_exists(filtered_df, 'overall_confidence'):
                    conf_by_model = filtered_df.groupby('model_category')['overall_confidence'].agg(['mean', 'std', 'count']).reset_index()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=conf_by_model['model_category'],
                        y=conf_by_model['mean'],
                        error_y=dict(type='data', array=conf_by_model['std']),
                        text=conf_by_model['mean'],
                        texttemplate='%{text:.3f}',
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title="Average Confidence by Model Category",
                        xaxis_title="Model Category",
                        yaxis_title="Average Confidence",
                        height=400,
                        yaxis_range=[0, 1.1]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Confidence data not available")
            
            # Show detailed statistics
            st.dataframe(model_perf, use_container_width=True)
        else:
            st.info("No model category data available for analysis")
    else:
        st.info("Model category data not available")
    
    # B3: Uncertainty Calibration Metrics
    st.subheader("3️⃣ Uncertainty Calibration Metrics")
    
    calibration = calculate_calibration_metrics(filtered_df)
    
    if calibration:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Calibration curve
            bin_data = pd.DataFrame(calibration['bin_data'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bin_data['confidence'],
                y=bin_data['accuracy'],
                mode='markers+lines',
                name='Calibration',
                marker=dict(size=bin_data['count'] / bin_data['count'].max() * 30),
                hovertemplate='Confidence: %{x:.2f}<br>Accuracy: %{y:.2f}<br>Count: %{marker.size:.0f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(dash='dash', color='gray')
            ))
            fig.update_layout(
                title="Calibration Curve",
                xaxis_title="Predicted Confidence",
                yaxis_title="Actual Accuracy",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Calibration Metrics:**")
            st.metric("ECE (Expected Calibration Error)", f"{calibration['ece']:.4f}")
            st.metric("MCE (Max Calibration Error)", f"{calibration['mce']:.4f}")
            st.metric("Brier Score", f"{calibration['brier_score']:.4f}")
            
            st.info("Lower values indicate better calibration")
    else:
        st.info("Insufficient data for calibration analysis")
    
    # ===== SECTION C: CRITICAL COMPARISONS WITH STATISTICAL TESTS =====
    st.markdown("---")
    st.header("📈 Critical Comparisons & Statistical Tests")
    
    # C1: Uncertainty-aware vs Baseline
    st.subheader("1️⃣ Uncertainty-Aware vs Baseline Experiments")
    
    if safe_column_exists(filtered_df, 'experiment_group') and safe_column_exists(filtered_df, 'overall_confidence'):
        groups = filtered_df['experiment_group'].unique()
        
        if len(groups) >= 2:
            # Let user select two groups to compare
            col1, col2 = st.columns(2)
            with col1:
                group1_name = st.selectbox("Select Group 1", options=groups, index=0)
            with col2:
                group2_name = st.selectbox("Select Group 2", options=groups, index=min(1, len(groups)-1))
            
            if group1_name != group2_name:
                group1_data = filtered_df[filtered_df['experiment_group'] == group1_name]
                group2_data = filtered_df[filtered_df['experiment_group'] == group2_name]
                
                # Comparison metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**{group1_name}**")
                    if safe_column_exists(group1_data, 'success'):
                        sr1 = group1_data['success'].mean() * 100
                        st.metric("Success Rate", f"{sr1:.1f}%")
                    conf1 = group1_data['overall_confidence'].mean()
                    st.metric("Avg Confidence", f"{conf1:.3f}")
                    st.metric("N", len(group1_data))
                
                with col2:
                    st.markdown(f"**{group2_name}**")
                    if safe_column_exists(group2_data, 'success'):
                        sr2 = group2_data['success'].mean() * 100
                        st.metric("Success Rate", f"{sr2:.1f}%")
                    conf2 = group2_data['overall_confidence'].mean()
                    st.metric("Avg Confidence", f"{conf2:.3f}")
                    st.metric("N", len(group2_data))
                
                with col3:
                    st.markdown("**Statistical Test**")
                    
                    # T-test on confidence
                    stat, p_val, test_type = perform_ttest(
                        group1_data['overall_confidence'],
                        group2_data['overall_confidence']
                    )
                    
                    st.metric("Test Type", test_type)
                    st.metric("p-value", f"{p_val:.4f}" if not np.isnan(p_val) else "N/A")
                    st.metric("Significance", get_significance_marker(p_val))
                
                # Visualization
                comparison_data = pd.concat([
                    group1_data[['overall_confidence']].assign(group=group1_name),
                    group2_data[['overall_confidence']].assign(group=group2_name)
                ])
                
                fig = px.box(
                    comparison_data,
                    x='group',
                    y='overall_confidence',
                    title=f"Confidence Distribution: {group1_name} vs {group2_name}",
                    labels={'group': 'Experiment Group', 'overall_confidence': 'Overall Confidence'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 experiment groups for comparison")
    else:
        st.info("Experiment group or confidence data not available")
    
    # C2: Large VLM vs SVLM
    st.subheader("2️⃣ Large VLM vs SVLM Performance")
    
    if safe_column_exists(filtered_df, 'model_category') and safe_column_exists(filtered_df, 'overall_confidence'):
        model_cats = filtered_df['model_category'].unique()
        
        if len(model_cats) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                model1 = st.selectbox("Select Model Category 1", options=model_cats, index=0, key='model1')
            with col2:
                model2 = st.selectbox("Select Model Category 2", options=model_cats, index=min(1, len(model_cats)-1), key='model2')
            
            if model1 != model2:
                model1_data = filtered_df[filtered_df['model_category'] == model1]
                model2_data = filtered_df[filtered_df['model_category'] == model2]
                
                # Comparison
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**{model1}**")
                    if safe_column_exists(model1_data, 'success'):
                        sr1 = model1_data['success'].mean() * 100
                        st.metric("Success Rate", f"{sr1:.1f}%")
                    conf1 = model1_data['overall_confidence'].mean()
                    st.metric("Avg Confidence", f"{conf1:.3f}")
                    st.metric("N", len(model1_data))
                
                with col2:
                    st.markdown(f"**{model2}**")
                    if safe_column_exists(model2_data, 'success'):
                        sr2 = model2_data['success'].mean() * 100
                        st.metric("Success Rate", f"{sr2:.1f}%")
                    conf2 = model2_data['overall_confidence'].mean()
                    st.metric("Avg Confidence", f"{conf2:.3f}")
                    st.metric("N", len(model2_data))
                
                with col3:
                    st.markdown("**Statistical Test**")
                    stat, p_val, test_type = perform_ttest(
                        model1_data['overall_confidence'],
                        model2_data['overall_confidence']
                    )
                    st.metric("Test Type", test_type)
                    st.metric("p-value", f"{p_val:.4f}" if not np.isnan(p_val) else "N/A")
                    st.metric("Significance", get_significance_marker(p_val))
                
                # Visualization: Side-by-side comparison
                fig = make_subplots(rows=1, cols=2, subplot_titles=['Success Rate', 'Confidence Distribution'])
                
                # Success rate comparison
                if safe_column_exists(filtered_df, 'success'):
                    sr_data = pd.DataFrame({
                        'Model': [model1, model2],
                        'Success Rate': [
                            model1_data['success'].mean(),
                            model2_data['success'].mean()
                        ]
                    })
                    fig.add_trace(
                        go.Bar(x=sr_data['Model'], y=sr_data['Success Rate'], name='Success Rate'),
                        row=1, col=1
                    )
                
                # Confidence distribution
                fig.add_trace(
                    go.Box(y=model1_data['overall_confidence'], name=model1),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Box(y=model2_data['overall_confidence'], name=model2),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 model categories for comparison")
    else:
        st.info("Model category or confidence data not available")
    
    # C3: Different Prompt Types (ANOVA)
    st.subheader("3️⃣ Prompt Type Comparison (ANOVA)")
    
    if safe_column_exists(filtered_df, 'prompt_type') and safe_column_exists(filtered_df, 'overall_confidence'):
        prompt_types = filtered_df['prompt_type'].dropna().unique()
        
        if len(prompt_types) >= 2:
            # Prepare data for ANOVA
            groups = [filtered_df[filtered_df['prompt_type'] == pt]['overall_confidence'] for pt in prompt_types]
            
            f_stat, p_value, posthoc_df = perform_anova(groups, prompt_types.tolist())
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**ANOVA Results**")
                st.metric("F-statistic", f"{f_stat:.3f}" if not np.isnan(f_stat) else "N/A")
                st.metric("p-value", f"{p_value:.4f}" if not np.isnan(p_value) else "N/A")
                st.metric("Significance", get_significance_marker(p_value))
                
                if not np.isnan(p_value):
                    if p_value < 0.05:
                        st.success("Significant difference detected among groups!")
                    else:
                        st.info("No significant difference among groups")
            
            with col2:
                if not posthoc_df.empty:
                    st.markdown("**Post-hoc Pairwise Comparisons (Bonferroni-corrected)**")
                    posthoc_display = posthoc_df.copy()
                    posthoc_display['p_value'] = posthoc_display['p_value'].apply(lambda x: f"{x:.4f}")
                    posthoc_display['significant'] = posthoc_display['significant'].apply(lambda x: "Yes ✓" if x else "No")
                    st.dataframe(posthoc_display, use_container_width=True)
            
            # Visualization
            fig = px.box(
                filtered_df,
                x='prompt_type',
                y='overall_confidence',
                title="Confidence Distribution by Prompt Type",
                labels={'prompt_type': 'Prompt Type', 'overall_confidence': 'Overall Confidence'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 prompt types for ANOVA")
    else:
        st.info("Prompt type or confidence data not available")
    
    # ===== SECTION D: DETAILED UNCERTAINTY METRICS =====
    st.markdown("---")
    st.header("🔬 Detailed Uncertainty Metrics")
    
    # Stage-wise confidence and entropy
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confidence by Stage")
        stage_confidence = []
        for stage in ['grounder', 'planner', 'ranker']:
            col = f'{stage}_confidence'
            if safe_column_exists(filtered_df, col):
                avg = filtered_df[col].mean()
                std = filtered_df[col].std()
                stage_confidence.append({'Stage': stage.capitalize(), 'Mean': avg, 'Std': std})
        
        if stage_confidence:
            stage_df = pd.DataFrame(stage_confidence)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=stage_df['Stage'],
                y=stage_df['Mean'],
                error_y=dict(type='data', array=stage_df['Std']),
                text=stage_df['Mean'],
                texttemplate='%{text:.3f}',
                textposition='outside'
            ))
            fig.update_layout(
                title="Average Confidence by Pipeline Stage",
                xaxis_title="Stage",
                yaxis_title="Confidence",
                height=400,
                yaxis_range=[0, 1.1]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Stage-wise confidence data not available")
    
    with col2:
        st.subheader("Entropy by Stage")
        stage_entropy = []
        for stage in ['grounder', 'planner', 'ranker']:
            col = f'{stage}_entropy'
            if safe_column_exists(filtered_df, col):
                avg = filtered_df[col].mean()
                std = filtered_df[col].std()
                stage_entropy.append({'Stage': stage.capitalize(), 'Mean': avg, 'Std': std})
        
        if stage_entropy:
            entropy_df = pd.DataFrame(stage_entropy)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=entropy_df['Stage'],
                y=entropy_df['Mean'],
                error_y=dict(type='data', array=entropy_df['Std']),
                text=entropy_df['Mean'],
                texttemplate='%{text:.3f}',
                textposition='outside'
            ))
            fig.update_layout(
                title="Average Entropy by Pipeline Stage",
                xaxis_title="Stage",
                yaxis_title="Entropy",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Stage-wise entropy data not available")
    
    # Confidence vs Entropy scatter
    if safe_column_exists(filtered_df, 'overall_confidence'):
        # Calculate overall entropy
        entropy_cols = [col for col in filtered_df.columns if '_entropy' in col and '_std' not in col]
        if entropy_cols:
            filtered_df['overall_entropy'] = filtered_df[entropy_cols].mean(axis=1)
            
            if safe_column_exists(filtered_df, 'overall_entropy'):
                fig = px.scatter(
                    filtered_df,
                    x='overall_confidence',
                    y='overall_entropy',
                    color='experiment_group' if safe_column_exists(filtered_df, 'experiment_group') else None,
                    hover_data=['experiment_id'],
                    title="Confidence vs Entropy Relationship",
                    labels={'overall_confidence': 'Overall Confidence', 'overall_entropy': 'Overall Entropy'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # ===== SECTION E: INDIVIDUAL EXPERIMENT EXPLORER =====
    st.markdown("---")
    st.header("🔍 Individual Experiment Explorer")
    
    # Select columns to display
    display_cols = ['experiment_id', 'experiment_group', 'timestamp', 'success', 'overall_confidence']
    display_cols += [col for col in ['prompt_type', 'model_category', 'batch_id', 'attempts', 'n_objects', 'query'] if col in filtered_df.columns]
    
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    
    # Sort by timestamp if available, otherwise by experiment_id
    if 'timestamp' in available_cols and safe_column_exists(filtered_df, 'timestamp'):
        sorted_df = filtered_df[available_cols].sort_values('timestamp', ascending=False)
    else:
        sorted_df = filtered_df[available_cols]
    
    st.dataframe(
        sorted_df,
        use_container_width=True,
        height=400
    )
    
    # Export functionality
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"uncertainty_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.metric("Filtered Experiments", len(filtered_df))
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard created for OWG Uncertainty Analysis Project*")

with tabs[4]:
    # --- Run Experiment ---
    st.header("🚀 Run Experiment")
    
    # Environment Settings
    st.subheader("🌍 Environment Settings")
    env_col1, env_col2 = st.columns(2)
    with env_col1:
        exp_seed = st.number_input("Random Seed", min_value=0, value=42, step=1, key="exp_seed")
    with env_col2:
        exp_n_objects = st.number_input("Number of Objects", min_value=1, max_value=20, value=12, step=1, key="exp_n_objects")
    
    st.markdown("---")
    
    # LiteLLM Status
    st.subheader("🤖 LiteLLM Status")

    litellm_col1, litellm_col2 = st.columns([2, 2])

    with litellm_col1:
        if st.button("🔍 Check LiteLLM Status", use_container_width=True):
            with st.spinner("Checking LiteLLM server..."):
                try:
                    result = check_litellm()   # <-- Clean call

                    if result["running"]:
                        st.session_state["litellm_running"] = True
                        st.success(f"✅ LiteLLM is running at: {result['endpoint']}")

                        # Handle model list
                        models = result["models"]
                        if models:
                            st.session_state["available_models"] = models
                            st.info(f"Found {len(models)} models")
                        else:
                            st.warning("Models endpoint returned no data.")
                            st.session_state["available_models"] = []

                    else:
                        st.session_state["litellm_running"] = False
                        st.session_state["available_models"] = []
                        st.error(result["error"])

                except Exception as e:
                    st.session_state["litellm_running"] = False
                    st.error(f"❌ Error checking LiteLLM: {e}")
    
    with litellm_col2:
    #     if st.button("▶️ Start LiteLLM", use_container_width=True):
    #         with st.spinner("Starting LiteLLM server..."):
    #             try:
    #                 import subprocess
    #                 litellm_config_path = "config/litellm/config.yaml"
    #                 if os.path.exists(litellm_config_path):
    #                     subprocess.Popen(["litellm", "--config", litellm_config_path], 
    #                                    stdout=subprocess.DEVNULL, 
    #                                    stderr=subprocess.DEVNULL)
    #                     time.sleep(3)
    #                     st.success("✅ LiteLLM started (check status)")
    #                     st.rerun()
    #                 else:
    #                     st.error(f"❌ Config file not found: {litellm_config_path}")
    #             except Exception as e:
    #                 st.error(f"❌ Failed to start LiteLLM: {e}")
    
    # with litellm_col3:
        status_indicator = "🟢" if st.session_state.get('litellm_running', False) else "🔴"
        st.metric("Status", status_indicator)
    
    # Display available models
    if st.session_state.get('available_models'):
        st.info(f"📋 Available models: {', '.join(st.session_state['available_models'][:5])}" + 
                (f" (+{len(st.session_state['available_models'])-5} more)" if len(st.session_state['available_models']) > 5 else ""))
    
    st.markdown("---")
    
    # User Query
    st.subheader("💬 User Query")
    user_query = st.text_input("Query for the robot", value="pick up the smallest object", key="user_query")
    
    st.markdown("---")
    
    # Configuration Editor
    st.subheader("⚙️ Pipeline Configuration")
    
    # Initialize session state for config
    if 'exp_config' not in st.session_state:
        st.session_state['exp_config'] = {
            'image_size_h': 448,
            'image_size_w': 448,
            'grounding': {'enabled': True},
            'planning': {'enabled': True},
            'grasping': {'enabled': True}
        }
    
    # Get available prompts (both system and user-defined)
    system_prompts = [f.replace(".txt", "") for f in os.listdir(default_prompt_lib.prompt_dir) if f.endswith(".txt")]
    user_prompts = [f.replace(".txt", "") for f in os.listdir(USER_PROMPT_DIR) if f.endswith(".txt")] if os.path.exists(USER_PROMPT_DIR) else []
    all_prompts = sorted(system_prompts) + sorted([f"👤 {p}" for p in user_prompts])
    
    available_models = st.session_state.get('available_models', ['gpt-4o', 'gpt-4o-mini', 'claude-3-5-sonnet-20241022'])
    
    # Helper function to create stage config UI
    def create_stage_config(stage_name, stage_key, default_config):
        with st.expander(f"🔧 {stage_name} Configuration", expanded=False):
            enabled = st.checkbox(f"Enable {stage_name}", value=True, key=f"{stage_key}_enabled")
            st.session_state['exp_config'][stage_key]['enabled'] = enabled
            
            if enabled:
                config_col1, config_col2 = st.columns(2)
                
                with config_col1:
                    st.markdown("**Prompt Settings**")
                    
                    # Prompt selection - full filename without extension
                    default_prompt = default_config['prompt_name']
                    try:
                        default_index = all_prompts.index(default_prompt) if default_prompt in all_prompts else 0
                    except:
                        # Check with user prefix
                        user_prefixed = f"👤 {default_prompt}"
                        default_index = all_prompts.index(user_prefixed) if user_prefixed in all_prompts else 0
                    
                    if stage_key == "grounding":
                        allowed_prefix = "referring_segmentation_"
                    elif stage_key == "planning":
                        allowed_prefix = "grasp_planning_"
                    elif stage_key == "grasping":
                        allowed_prefix = "grasp_ranking_"
                    else:
                        allowed_prefix = ""

                    # ---------------------------------------------------------
                    # 🔹 Filter system prompts (must match prefix)
                    # 🔹 User prompts are NOT filtered — all included
                    # ---------------------------------------------------------
                    filtered_system_prompts = [
                        p for p in system_prompts if p.startswith(allowed_prefix)
                    ]

                    # user prompts remain unfiltered
                    filtered_user_prompts = user_prompts[:]   # copy

                    # Build UI list
                    filtered_all_prompts = (
                        sorted(filtered_system_prompts) +
                        sorted([f"👤 {p}" for p in filtered_user_prompts])
                    )

                    if not filtered_system_prompts:
                        st.warning(f"No **system prompts** found starting with `{allowed_prefix}`.")
                        # still show user prompts if available
                        if not filtered_user_prompts:
                            st.error("No prompts available at all.")
                            filtered_all_prompts = ["<no prompts available>"]

                    # ---------------------------------------------------------
                    # 🎛️ Selectbox for filtered prompts
                    # ---------------------------------------------------------
                    default_prompt = default_config["prompt_name"]

                    try:
                        default_index = filtered_all_prompts.index(default_prompt)
                    except:
                        user_prefixed = f"👤 {default_prompt}"
                        default_index = (
                            filtered_all_prompts.index(user_prefixed)
                            if user_prefixed in filtered_all_prompts
                            else 0
                        )

                    prompt_name = st.selectbox(
                        "Prompt File",
                        options=filtered_all_prompts, #all_prompts,
                        index=default_index,
                        key=f"{stage_key}_prompt_name",
                        help="Select the complete prompt file (full name without .txt)"
                    )
                    
                    # Determine prompt_root_dir based on selection
                    is_user_prompt = prompt_name.startswith("👤")
                    prompt_root_dir = USER_PROMPT_DIR if is_user_prompt else UNCERTAINTY_DIR
                    clean_prompt_name = prompt_name.replace("👤 ", "")
                    
                    # Prompt template
                    prompt_template = st.text_area(
                        "Prompt Template",
                        value=default_config.get('prompt_template', ''),
                        height=100,
                        key=f"{stage_key}_template",
                        help="Use {user_input} as placeholder for the user query"
                    )
                    
                    # Preview prompt button
                    if st.button(f"👁️ Preview Prompt", key=f"{stage_key}_preview"):
                        filepath = os.path.join(prompt_root_dir, f"{clean_prompt_name}.txt")
                        
                        st.markdown("**📝 Prompt Preview:**")
                        
                        # Show the prompt template with user_input filled first
                        if prompt_template:
                            try:
                                filled_template = prompt_template.format(user_input=user_query)
                                st.markdown("**User Prompt (with query):**")
                                st.info(filled_template)
                            except Exception as e:
                                st.warning(f"Could not format template: {e}")
                            
                            st.markdown("---")
                        
                        # Show system prompt
                        if os.path.exists(filepath):
                            with open(filepath, "r") as f:
                                content = f.read()
                            
                            st.markdown(f"**System Prompt: `{clean_prompt_name}.txt`** from `{prompt_root_dir}`")
                            st.code(content, language="text")
                        else:
                            st.error(f"⚠️ File not found: {clean_prompt_name}.txt in {prompt_root_dir}")
                
                with config_col2:
                    st.markdown("**Model Settings**")
                    
                    model_name = st.selectbox(
                        "Model",
                        options=available_models,
                        index=available_models.index(default_config['request']['model_name']) if default_config['request']['model_name'] in available_models else 0,
                        key=f"{stage_key}_model"
                    )
                    
                    temperature = st.slider(
                        "Temperature",
                        min_value=0.0,
                        max_value=2.0,
                        value=float(default_config['request'].get('temperature', 0.1)),
                        step=0.1,
                        key=f"{stage_key}_temp"
                    )
                    
                    n = st.number_input(
                        "Number of Completions (n)",
                        min_value=1,
                        max_value=10,
                        value=default_config['request'].get('n', 2),
                        step=1,
                        key=f"{stage_key}_n"
                    )
                    
                    max_tokens = st.number_input(
                        "Max Tokens",
                        min_value=64,
                        max_value=16384,
                        value=default_config['request'].get('max_tokens', 512),
                        step=64,
                        key=f"{stage_key}_tokens"
                    )
                    
                    logprobs = st.checkbox(
                        "Enable Logprobs",
                        value=default_config['request'].get('logprobs', True),
                        key=f"{stage_key}_logprobs"
                    )
                
                # Store config in session state with prompt_root_dir
                st.session_state['exp_config'][stage_key].update({
                    'prompt_name': clean_prompt_name,
                    'prompt_template': prompt_template,
                    'prompt_root_dir': prompt_root_dir,
                    'request': {
                        'model_name': model_name,
                        'temperature': temperature,
                        'n': n,
                        'max_tokens': max_tokens,
                        'logprobs': logprobs
                    }
                })
    
    # Create configs for each stage with prompt_root_dir
    grounding_default = {
        'prompt_name': 'referring_segmentation_cautious',
        'prompt_template': 'Description: {user_input}',
        'prompt_root_dir': UNCERTAINTY_DIR,
        'request': {'model_name': 'gpt-4o', 'temperature': 0.1, 'n': 2, 'max_tokens': 256, 'logprobs': True}
    }
    
    planning_default = {
        'prompt_name': 'grasp_planning_confidence',
        'prompt_template': 'Task instruction: "Target object {user_input}".',
        'prompt_root_dir': UNCERTAINTY_DIR,
        'request': {'model_name': 'gpt-4o', 'temperature': 0.0, 'n': 2, 'max_tokens': 256, 'logprobs': True}
    }
    
    grasping_default = {
        'prompt_name': 'grasp_ranking_uncertainty_description',
        'prompt_template': 'Rank the grasp poses.',
        'prompt_root_dir': UNCERTAINTY_DIR,
        'request': {'model_name': 'gpt-4o', 'temperature': 0.0, 'n': 2, 'max_tokens': 256, 'logprobs': True}
    }
    
    create_stage_config("Grounding", "grounding", grounding_default)
    create_stage_config("Planning", "planning", planning_default)
    create_stage_config("Grasping", "grasping", grasping_default)
    
    st.markdown("---")
    
    # Config Preview and Save
    preview_col, save_col = st.columns(2)
    
    with preview_col:
        if st.button("👁️ Preview Full Config YAML", use_container_width=True):
            st.session_state['show_config_preview'] = True
    
    with save_col:
        if st.button("💾 Save Config", use_container_width=True):
            import yaml
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_dir = "config/pyb/user_defined"  # ✅ Changed from OWG/config/...
            os.makedirs(config_dir, exist_ok=True)
            
            config_path = os.path.join(config_dir, f"config_{timestamp}.yaml")
            
            # Build full config with per-stage prompt_root_dir
            full_config = {
                'image_size_h': 448,
                'image_size_w': 448,
                'image_crop': None
            }
            
            # Map stage keys to config keys
            stage_to_config_key = {
                'grounding': 'grounding_prompt_root_dir',
                'planning': 'planning_prompt_root_dir',
                'grasping': 'grasping_prompt_root_dir'
            }
            
            for stage_key in ['grounding', 'planning', 'grasping']:
                if st.session_state['exp_config'][stage_key].get('enabled', True):
                    stage_config = st.session_state['exp_config'][stage_key].copy()
                    stage_config.pop('enabled', None)
                    
                    # Extract prompt_root_dir and convert to relative path
                    prompt_root_dir = stage_config.pop('prompt_root_dir', UNCERTAINTY_DIR)
                    # ✅ Convert absolute path to relative if needed
                    if prompt_root_dir.startswith('/home/owner/OWG/'):
                        prompt_root_dir = './' + prompt_root_dir.replace('/home/owner/OWG/', '')
                    elif not prompt_root_dir.startswith('./'):
                        prompt_root_dir = './' + prompt_root_dir.lstrip('/')
                    
                    config_key = stage_to_config_key[stage_key]
                    full_config[config_key] = prompt_root_dir
                    
                    # ✅ Add 'detail' to request
                    if 'request' in stage_config:
                        stage_config['request']['detail'] = 'auto'
                    
                    # Add default visualizer and other settings based on stage
                    if stage_key == 'grounding':
                        # ✅ Add seed to request
                        if 'request' in stage_config:
                            stage_config['request']['seed'] = 12
                        
                        stage_config.update({
                            'include_raw_image': True,
                            'use_subplot_prompt': False,
                            'subplot_size': 224,
                            'do_refine_marks': False,
                            'refine_marks': {
                                'maximum_hole_area': 0.01,
                                'maximum_island_area': 0.01,
                                'minimum_mask_area': 0.02,
                                'maximum_mask_area': 1.0
                            },
                            'do_inctx': False,
                            'inctx_prompt_name': None,
                            'visualizer': {
                                'label': {
                                    'text_include': True,
                                    'text_scale': 0.5,
                                    'text_thickness': 2,
                                    'text_padding': 2,
                                    'text_position': 'TOP_CENTER'
                                },
                                'box': {
                                    'box_include': False,
                                    'box_thickness': 2
                                },
                                'mask': {
                                    'mask_include': True,
                                    'mask_opacity': 0.25
                                },
                                'polygon': {
                                    'polygon_include': True,
                                    'polygon_thickness': 2
                                }
                            }
                        })
                    elif stage_key == 'planning':
                        stage_config.update({
                            'response_format': 'json',
                            'include_raw_image': False,
                            'use_subplot_prompt': False,
                            'subplot_size': 448,
                            'do_refine_marks': False,
                            'refine_marks': {
                                'maximum_hole_area': 0.01,
                                'maximum_island_area': 0.01,
                                'minimum_mask_area': 0.02,
                                'maximum_mask_area': 1.0
                            },
                            'do_inctx': False,
                            'inctx_prompt_name': 'pyb/inctx_grasp_planning.pt',
                            'visualizer': {
                                'label': {
                                    'text_include': True,
                                    'text_scale': 0.5,
                                    'text_thickness': 2,
                                    'text_padding': 2,
                                    'text_position': 'CENTER_OF_MASS'
                                },
                                'box': {
                                    'box_include': False,
                                    'box_thickness': 2
                                },
                                'mask': {
                                    'mask_include': True,
                                    'mask_opacity': 0.25
                                },
                                'polygon': {
                                    'polygon_include': True,
                                    'polygon_thickness': 1
                                }
                            }
                        })
                    else:  # grasping
                        stage_config.update({
                            'crop_square_size': 196,
                            'use_3d_prompt': False,
                            'gripper_mesh_path': 'owg_robot/assets/robotiq_2f_140/robotiq_arg2f_140.obj',
                            'use_subplot_prompt': True,
                            'subplot_size': 224,
                            'do_inctx': False,
                            'inctx_prompt_name': 'pyb/inctx_grasp_ranking.pt',
                            'visualizer': {
                                'as_line': True,
                                'line_thickness': 8,
                                'grasp_colors': 'RED,GREEN',
                                'with_gray': False,
                                'label': {
                                    'label_include': False,
                                    'text_color': 'WHITE',
                                    'text_rect_color': 'BLACK',
                                    'text_padding': 2,
                                    'text_thickness': 1,
                                    'text_scale': 0.7,
                                    'text_position': 'CENTER'
                                },
                                'box': {
                                    'box_include': False,
                                    'box_color': 'RED',
                                    'box_thickness': 2
                                },
                                'mask': {
                                    'mask_include': False,
                                    'mask_color': 'RED',
                                    'mask_opacity': 0.15
                                },
                                'polygon': {
                                    'polygon_include': True,
                                    'polygon_color': 'RED',
                                    'polygon_thickness': 2
                                }
                            }
                        })
                    
                    full_config[stage_key] = stage_config
            
            with open(config_path, 'w') as f:
                yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)
            
            st.success(f"✅ Config saved to: {config_path}")
            
            # Show which directories are being used
            dir_info = []
            for stage_key in ['grounding', 'planning', 'grasping']:
                if st.session_state['exp_config'][stage_key].get('enabled'):
                    stage_name = stage_key.capitalize()
                    prompt_dir = st.session_state['exp_config'][stage_key].get('prompt_root_dir', 'N/A')
                    dir_info.append(f"**{stage_name}**: `{prompt_dir}`")
            
            if dir_info:
                st.info("📁 **Prompt directories:**\n\n" + "\n\n".join(dir_info))
            
            st.session_state['last_saved_config'] = config_path

    # Show config preview
    if st.session_state.get('show_config_preview', False):
        st.markdown("### 📄 Config Preview")
        
        import yaml
        preview_config = {
            'image_size_h': 448,
            'image_size_w': 448,
            'image_crop': None
        }
        
        # Add per-stage prompt_root_dir
        stage_to_config_key = {
            'grounding': 'grounding_prompt_root_dir',
            'planning': 'planning_prompt_root_dir',
            'grasping': 'grasping_prompt_root_dir'
        }
        
        for stage_key in ['grounding', 'planning', 'grasping']:
            if st.session_state['exp_config'][stage_key].get('enabled', True):
                stage_config = {k: v for k, v in st.session_state['exp_config'][stage_key].items() if k not in ['enabled', 'prompt_root_dir']}
                preview_config[stage_key] = stage_config
                
                # Add prompt_root_dir to top level
                prompt_root_dir = st.session_state['exp_config'][stage_key].get('prompt_root_dir', UNCERTAINTY_DIR)
                config_key = stage_to_config_key[stage_key]
                preview_config[config_key] = prompt_root_dir
        
        st.code(yaml.dump(preview_config, default_flow_style=False, sort_keys=False), language='yaml')
        
        if st.button("❌ Close Preview"):
            st.session_state['show_config_preview'] = False
            st.rerun()
    
    st.markdown("---")
    
    # Run Experiment Button
    st.subheader("🎯 Execute Pipeline")
    
    run_col1, run_col2 = st.columns([3, 1])
    with run_col1:
        if st.button("▶️ Run Experiment Pipeline", use_container_width=True, type="primary"):
            if not st.session_state.get('litellm_running', False):
                st.error("❌ LiteLLM is not running. Please start it first.")
            else:
                with st.spinner("Running experiment pipeline..."):
                    try:
                        import subprocess
                        
                        # Save config first if not already saved
                        if 'last_saved_config' not in st.session_state:
                            from datetime import datetime
                            import yaml
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            config_dir = "config/pyb/user_defined"
                            os.makedirs(config_dir, exist_ok=True)
                            config_path = os.path.join(config_dir, f"config_{timestamp}.yaml")
                            
                            full_config = {'image_size_h': 448, 'image_size_w': 448, 'image_crop': None}
                            
                            stage_to_config_key = {
                                'grounding': 'grounding_prompt_root_dir',
                                'planning': 'planning_prompt_root_dir',
                                'grasping': 'grasping_prompt_root_dir'
                            }
                            
                            for stage_key in ['grounding', 'planning', 'grasping']:
                                if st.session_state['exp_config'][stage_key].get('enabled', True):
                                    stage_config = {k: v for k, v in st.session_state['exp_config'][stage_key].items() if k not in ['enabled', 'prompt_root_dir']}
                                    full_config[stage_key] = stage_config
                                    
                                    # Add prompt_root_dir to top level
                                    prompt_root_dir = st.session_state['exp_config'][stage_key].get('prompt_root_dir', UNCERTAINTY_DIR)
                                    config_key = stage_to_config_key[stage_key]
                                    full_config[config_key] = prompt_root_dir
                            
                            with open(config_path, 'w') as f:
                                yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)
                            
                            st.session_state['last_saved_config'] = config_path
                        
                        # Run the pipeline
                        cmd = [
                            "python", "notebooks/owg_evaluation_pipeline.py",
                            "--seed", str(exp_seed),
                            "--config", st.session_state['last_saved_config'],
                            "--query", user_query,
                            "--n-objects", str(exp_n_objects),
                            "--output-dir", "output"
                        ]
                        
                        st.info(f"**Running command:**")
                        st.code(" ".join(f'"{arg}"' if " " in arg else arg for arg in cmd), language="bash")
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                        
                        if result.returncode == 0:
                            st.success("✅ Experiment completed successfully!")
                            st.text_area("Output:", result.stdout, height=300)
                        else:
                            st.error("❌ Experiment failed!")
                            st.text_area("Error output:", result.stderr, height=300)
                    
                    except subprocess.TimeoutExpired:
                        st.error("❌ Experiment timed out (5 minutes)")
                    except Exception as e:
                        st.error(f"❌ Error running experiment: {e}")
    
    with run_col2:
        if st.button("🔄 Restart PyBullet", use_container_width=True):
            st.info("PyBullet will be restarted when running the experiment")