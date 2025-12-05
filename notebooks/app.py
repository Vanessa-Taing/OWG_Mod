# app.py
import sys, os
import requests
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import json
import os
import pandas as pd
from datetime import datetime
import time
import altair as alt
from owg_mod.prompt_library import SystemPromptLibrary
from owg_mod.model_utils_litellm import check_litellm

# Paths for metrics
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_PATH = os.path.join(BASE_DIR, "logs", "litellm_logs.jsonl")
METRICS_PATH = os.path.join(BASE_DIR, "logs", "experiment_metrics.jsonl")
LOG_PATH_UNCERT = os.path.join(BASE_DIR, "logs", "uncertainty_logs.jsonl")

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

# ADD this helper function before the tabs section (around line 60):
def calculate_calibration_metrics(uncertainty_logs, grasp_logs):
    """Calculate Expected Calibration Error from linked logs"""
    # Link logs by experiment_id
    merged = []
    
    for u_log in uncertainty_logs:
        exp_id = u_log.get("experiment_id")
        if not exp_id:
            continue
            
        # Find matching grasp logs
        matching_grasps = [g for g in grasp_logs if g.get("experiment_id") == exp_id]
        
        for grasp in matching_grasps:
            # Extract confidence from uncertainty log
            planner_meta = u_log.get("metadata", {}).get("planner", [])
            if planner_meta:
                confidence = planner_meta[0].get("confidence", -1)
                if confidence != -1:
                    merged.append({
                        "predicted_confidence": confidence,
                        "actual_success": 1 if grasp["success"] else 0
                    })
    
    if not merged:
        return None, None
    
    df = pd.DataFrame(merged)
    
    # Calculate ECE (Expected Calibration Error)
    bins = np.linspace(0, 1, 11)  # 10 bins
    df['bin'] = pd.cut(df['predicted_confidence'], bins=bins, include_lowest=True)
    
    calibration_data = df.groupby('bin', observed=True).agg({
        'predicted_confidence': 'mean',
        'actual_success': 'mean'
    }).dropna()
    
    # ECE = average absolute difference between confidence and accuracy
    if len(calibration_data) > 0:
        ece = np.abs(calibration_data['predicted_confidence'] - calibration_data['actual_success']).mean()
    else:
        ece = None
    
    return calibration_data, ece

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
    st.header("🧠 Uncertainty Analysis")
    
    # ========== LOAD DATA ==========
    uncertainty_logs_exist = os.path.exists(LOG_PATH_UNCERT)
    grasp_logs_exist = os.path.exists(METRICS_PATH)
    
    if not uncertainty_logs_exist:
        st.info("📭 No uncertainty logs found. Run experiments to generate data.")
        st.code("python notebooks/owg_evaluation_pipeline.py --seed 42 --query 'pick smallest' --n-objects 12")
    else:
        # Load logs
        with open(LOG_PATH_UNCERT, "r") as f:
            uncertainty_logs = [json.loads(line) for line in f if line.strip()]
        
        grasp_logs = []
        if grasp_logs_exist:
            with open(METRICS_PATH, "r") as f:
                grasp_logs = [json.loads(line) for line in f if line.strip()]
        
        if not uncertainty_logs:
            st.info("📭 Empty uncertainty logs.")
        else:
            # ========== SIDEBAR FILTERS ==========
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 🔍 Uncertainty Filters")
                
                # Extract unique values
                all_exp_groups = sorted(set(log.get("experiment_group", "unknown") for log in uncertainty_logs))
                all_exp_ids = sorted(set(log.get("experiment_id", "unknown") for log in uncertainty_logs))
                
                # Batch experiment detection
                batch_experiments = {}
                for exp_id in all_exp_ids:
                    # Extract batch name from experiment_id pattern: YYYYMMDD_hash_seed
                    parts = exp_id.split('_')
                    if len(parts) >= 2:
                        batch_key = f"{parts[0]}_{parts[1]}"  # date_hash
                        if batch_key not in batch_experiments:
                            batch_experiments[batch_key] = []
                        batch_experiments[batch_key].append(exp_id)
                
                # Batch filter
                st.markdown("#### 📦 Filter by Batch")
                if len(batch_experiments) > 1:
                    batch_options = ["All Batches"] + list(batch_experiments.keys())
                    selected_batch = st.selectbox(
                        "Select batch experiment",
                        options=batch_options,
                        help="Experiments grouped by date and config"
                    )
                    
                    if selected_batch != "All Batches":
                        filtered_exp_ids = batch_experiments[selected_batch]
                        st.info(f"📊 {len(filtered_exp_ids)} experiments in this batch")
                    else:
                        filtered_exp_ids = all_exp_ids
                else:
                    st.info(f"📊 {len(all_exp_ids)} total experiments")
                    filtered_exp_ids = all_exp_ids
                
                # Experiment group filter
                st.markdown("#### 🏷️ Experiment Type")
                selected_groups = st.multiselect(
                    "Experiment Group",
                    options=all_exp_groups,
                    default=all_exp_groups,
                    help="Baseline vs uncertainty-aware prompts"
                )
                
                # Individual experiment filter
                st.markdown("#### 🔬 Individual Experiments")
                available_exp_ids = [eid for eid in filtered_exp_ids 
                                    if any(log.get("experiment_id") == eid and 
                                          log.get("experiment_group") in selected_groups 
                                          for log in uncertainty_logs)]
                
                if len(available_exp_ids) > 10:
                    use_all = st.checkbox("Select all experiments", value=True)
                    if use_all:
                        selected_ids = available_exp_ids
                    else:
                        selected_ids = st.multiselect(
                            "Select specific experiments",
                            options=available_exp_ids,
                            default=available_exp_ids[:5],
                            help="Choose individual experiment IDs"
                        )
                else:
                    selected_ids = st.multiselect(
                        "Select experiments",
                        options=available_exp_ids,
                        default=available_exp_ids,
                        help="Choose experiment IDs to analyze"
                    )
                
                st.markdown("---")
            
            # Filter logs based on selections
            filtered_uncertainty = [
                log for log in uncertainty_logs
                if log.get("experiment_group") in selected_groups
                and log.get("experiment_id") in selected_ids
            ]
            
            filtered_grasp = [
                log for log in grasp_logs
                if log.get("experiment_group") in selected_groups
                and log.get("experiment_id") in selected_ids
            ]
            
            if not filtered_uncertainty:
                st.warning("⚠️ No logs match your filters. Adjust sidebar settings.")
            else:
                # ========== OVERVIEW METRICS ==========
                st.subheader("📊 Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                baseline_count = sum(1 for log in filtered_uncertainty if log.get("experiment_group") == "baseline")
                uncertainty_count = sum(1 for log in filtered_uncertainty if log.get("experiment_group") == "uncertainty_aware")
                
                col1.metric("Total Experiments", len(filtered_uncertainty))
                col2.metric("Baseline Runs", baseline_count)
                col3.metric("Uncertainty-Aware", uncertainty_count)
                col4.metric("Grasp Attempts", len(filtered_grasp) if filtered_grasp else "N/A")
                
                # Success rate comparison if grasp data available
                if filtered_grasp:
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    
                    total_success = sum(1 for g in filtered_grasp if g.get("success"))
                    total_grasps = len(filtered_grasp)
                    overall_success = total_success / total_grasps * 100 if total_grasps > 0 else 0
                    
                    baseline_grasps = [g for g in filtered_grasp if g.get("experiment_group") == "baseline"]
                    uncertainty_grasps = [g for g in filtered_grasp if g.get("experiment_group") == "uncertainty_aware"]
                    
                    baseline_success = sum(1 for g in baseline_grasps if g.get("success")) / len(baseline_grasps) * 100 if baseline_grasps else 0
                    uncertainty_success = sum(1 for g in uncertainty_grasps if g.get("success")) / len(uncertainty_grasps) * 100 if uncertainty_grasps else 0
                    
                    col1.metric("Overall Success Rate", f"{overall_success:.1f}%")
                    col2.metric("Baseline Success", f"{baseline_success:.1f}%")
                    col3.metric("Uncertainty Success", f"{uncertainty_success:.1f}%", 
                               delta=f"{uncertainty_success - baseline_success:+.1f}%")
                
                st.markdown("---")
                
                # ========== ANALYSIS TABS ==========
                analysis_tabs = st.tabs([
                    "📈 Trends & Patterns",
                    "🎯 Calibration Analysis",
                    "📊 Statistical Tests",
                    "🏆 Leaderboard",
                    "📋 Raw Data"
                ])
                
                # ========== TAB 1: TRENDS & PATTERNS ==========
                with analysis_tabs[0]:
                    st.markdown("### Uncertainty Metrics Over Time")
                    
                    # Extract and normalize data
                    df_uncert = pd.json_normalize(filtered_uncertainty, sep="_")
                    
                    if "timestamp" in df_uncert.columns:
                        df_uncert["timestamp"] = pd.to_datetime(df_uncert["timestamp"], errors="coerce")
                    
                    # Helper function to extract entropy & confidence
                    def extract_values(metadata):
                        out = {}
                        if isinstance(metadata, dict):
                            for key, val in metadata.items():
                                if isinstance(val, list):
                                    entropies = [x.get("entropy") for x in val if isinstance(x, dict) and "entropy" in x]
                                    confidences = [
                                        (x.get("confidence") if x.get("confidence") != -1 else np.nan)
                                        for x in val if isinstance(x, dict) and "confidence" in x
                                    ]
                                    if entropies:
                                        out[f"{key}_entropy"] = np.mean(entropies)
                                    if confidences:
                                        out[f"{key}_confidence"] = np.nanmean(confidences)
                        return out
                    
                    # Extract metrics
                    metadata_cols = [c for c in df_uncert.columns if c.startswith("metadata_")]
                    all_rows = []
                    
                    for i, row in df_uncert.iterrows():
                        merged = {
                            "timestamp": row["timestamp"],
                            "experiment_group": row.get("experiment_group", "unknown")
                        }
                        for col in metadata_cols:
                            values = extract_values(row[col])
                            for k, v in values.items():
                                merged[f"{col.replace('metadata_', '')}_{k}"] = v
                        all_rows.append(merged)
                    
                    df_metrics = pd.DataFrame(all_rows)
                    
                    if not df_metrics.empty and "timestamp" in df_metrics.columns:
                        entropy_cols = [c for c in df_metrics.columns if c.endswith("_entropy")]
                        conf_cols = [c for c in df_metrics.columns if c.endswith("_confidence")]
                        
                        # Create visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🔵 Entropy Trends")
                            if entropy_cols:
                                melted_entropy = df_metrics.melt(
                                    id_vars=["timestamp", "experiment_group"],
                                    value_vars=entropy_cols,
                                    var_name="metric",
                                    value_name="value"
                                )
                                
                                chart_entropy = (
                                    alt.Chart(melted_entropy.dropna(subset=['value']))
                                    .mark_line(point=True, opacity=0.7)
                                    .encode(
                                        x=alt.X("timestamp:T", title="Time"),
                                        y=alt.Y("value:Q", title="Entropy (bits)", scale=alt.Scale(domain=[0, 1.5])),
                                        color=alt.Color("metric:N", legend=alt.Legend(title="Metric")),
                                        strokeDash=alt.StrokeDash("experiment_group:N", legend=alt.Legend(title="Group")),
                                        tooltip=["timestamp:T", "metric:N", "value:Q", "experiment_group:N"]
                                    )
                                    .properties(height=300)
                                    .interactive()
                                )
                                st.altair_chart(chart_entropy, use_container_width=True)
                                
                                # Summary stats
                                st.markdown("**Summary Statistics:**")
                                for col in entropy_cols:
                                    mean_val = df_metrics[col].mean()
                                    std_val = df_metrics[col].std()
                                    st.text(f"{col}: μ={mean_val:.3f}, σ={std_val:.3f}")
                            else:
                                st.info("No entropy data available")
                        
                        with col2:
                            st.markdown("#### 🟢 Confidence Trends")
                            if conf_cols:
                                melted_conf = df_metrics.melt(
                                    id_vars=["timestamp", "experiment_group"],
                                    value_vars=conf_cols,
                                    var_name="metric",
                                    value_name="value"
                                )
                                
                                chart_conf = (
                                    alt.Chart(melted_conf.dropna(subset=['value']))
                                    .mark_line(point=True, opacity=0.7)
                                    .encode(
                                        x=alt.X("timestamp:T", title="Time"),
                                        y=alt.Y("value:Q", title="Confidence", scale=alt.Scale(domain=[0, 1])),
                                        color=alt.Color("metric:N", legend=alt.Legend(title="Metric")),
                                        strokeDash=alt.StrokeDash("experiment_group:N", legend=alt.Legend(title="Group")),
                                        tooltip=["timestamp:T", "metric:N", "value:Q", "experiment_group:N"]
                                    )
                                    .properties(height=300)
                                    .interactive()
                                )
                                st.altair_chart(chart_conf, use_container_width=True)
                                
                                # Summary stats
                                st.markdown("**Summary Statistics:**")
                                for col in conf_cols:
                                    mean_val = df_metrics[col].mean()
                                    std_val = df_metrics[col].std()
                                    st.text(f"{col}: μ={mean_val:.3f}, σ={std_val:.3f}")
                            else:
                                st.info("No confidence data available")
                        
                        # Correlation heatmap
                        st.markdown("---")
                        st.markdown("### 🔗 Metric Correlations")
                        
                        corr_cols = entropy_cols + conf_cols
                        if len(corr_cols) >= 2:
                            corr_data = df_metrics[corr_cols].dropna()
                            if len(corr_data) > 1:
                                corr_matrix = corr_data.corr()
                                
                                corr_reset = corr_matrix.reset_index().melt(id_vars='index')
                                corr_reset.columns = ['Variable 1', 'Variable 2', 'Correlation']
                                
                                heatmap = alt.Chart(corr_reset).mark_rect().encode(
                                    x=alt.X('Variable 1:N', title=None, axis=alt.Axis(labelAngle=-45)),
                                    y=alt.Y('Variable 2:N', title=None),
                                    color=alt.Color('Correlation:Q', 
                                                   scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                                                   legend=alt.Legend(title="Correlation")),
                                    tooltip=['Variable 1', 'Variable 2', alt.Tooltip('Correlation:Q', format='.3f')]
                                ).properties(height=400)
                                
                                st.altair_chart(heatmap, use_container_width=True)
                    else:
                        st.info("Insufficient data for trend analysis")
                
                # ========== TAB 2: CALIBRATION ANALYSIS ==========
                with analysis_tabs[1]:
                    st.markdown("### 🎯 Expected Calibration Error (ECE)")
                    
                    if not filtered_grasp:
                        st.warning("⚠️ No grasp logs available. Calibration requires linked experiment outcomes.")
                        st.info("💡 **Tip:** Run batch experiments to generate sufficient data:\n"
                               "```bash\npython notebooks/batch_experiments.py --mode baseline_vs_uncertainty\n```")
                    else:
                        # Calculate calibration
                        calibration_data, ece = calculate_calibration_metrics(filtered_uncertainty, filtered_grasp)
                        
                        if calibration_data is not None and len(calibration_data) > 1:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if ece is not None:
                                    st.metric("ECE Score", f"{ece:.3f}",
                                             help="Expected Calibration Error: Lower is better (0 = perfect)")
                                    
                                    # Interpretation
                                    if ece < 0.05:
                                        st.success("✅ Excellent calibration")
                                    elif ece < 0.10:
                                        st.info("ℹ️ Good calibration")
                                    elif ece < 0.15:
                                        st.warning("⚠️ Moderate calibration")
                                    else:
                                        st.error("❌ Poor calibration")
                            
                            with col2:
                                st.metric("Calibration Bins", len(calibration_data),
                                         help="Number of confidence bins with data")
                            
                            with col3:
                                # Calculate calibration bias
                                avg_diff = (calibration_data['predicted_confidence'] - 
                                           calibration_data['actual_success']).mean()
                                
                                if abs(avg_diff) < 0.05:
                                    st.metric("Calibration Bias", "Well Calibrated ✓", 
                                             delta=f"{avg_diff:+.3f}")
                                elif avg_diff > 0:
                                    st.metric("Calibration Bias", "Overconfident", 
                                             delta=f"+{avg_diff:.3f}", delta_color="inverse")
                                else:
                                    st.metric("Calibration Bias", "Underconfident",
                                             delta=f"{avg_diff:.3f}", delta_color="normal")
                            
                            # Reliability diagram
                            st.markdown("---")
                            st.markdown("#### 📊 Reliability Diagram")
                            
                            calib_plot = calibration_data.reset_index()
                            calib_plot['bin_center'] = calib_plot['predicted_confidence']
                            
                            # Main calibration line
                            chart = alt.Chart(calib_plot).mark_line(
                                point=alt.OverlayMarkDef(size=100, filled=True),
                                strokeWidth=3
                            ).encode(
                                x=alt.X('bin_center:Q', 
                                       title='Predicted Confidence', 
                                       scale=alt.Scale(domain=[0, 1]),
                                       axis=alt.Axis(format='%')),
                                y=alt.Y('actual_success:Q', 
                                       title='Actual Success Rate', 
                                       scale=alt.Scale(domain=[0, 1]),
                                       axis=alt.Axis(format='%')),
                                tooltip=[
                                    alt.Tooltip('bin_center:Q', title='Confidence', format='.2%'),
                                    alt.Tooltip('actual_success:Q', title='Success Rate', format='.2%')
                                ]
                            ).properties(height=400)
                            
                            # Perfect calibration reference line
                            reference = alt.Chart(
                                pd.DataFrame({'x': [0, 1], 'y': [0, 1]})
                            ).mark_line(
                                strokeDash=[5, 5], 
                                color='gray',
                                strokeWidth=2
                            ).encode(
                                x='x:Q', 
                                y='y:Q'
                            )
                            
                            # Combine charts
                            final_chart = (reference + chart).configure_axis(
                                gridOpacity=0.3
                            ).configure_view(
                                strokeWidth=0
                            )
                            
                            st.altair_chart(final_chart, use_container_width=True)
                            
                            st.markdown("""
                            **How to interpret:**
                            - Points on the diagonal = perfect calibration
                            - Points above diagonal = model is overconfident
                            - Points below diagonal = model is underconfident
                            """)
                            
                            # Group comparison if both baseline and uncertainty available
                            baseline_u_logs = [log for log in filtered_uncertainty if log.get("experiment_group") == "baseline"]
                            uncertainty_u_logs = [log for log in filtered_uncertainty if log.get("experiment_group") == "uncertainty_aware"]
                            
                            if baseline_u_logs and uncertainty_u_logs:
                                st.markdown("---")
                                st.markdown("#### 📊 Calibration by Experiment Group")
                                
                                baseline_calib, baseline_ece = calculate_calibration_metrics(baseline_u_logs, filtered_grasp)
                                uncertainty_calib, uncertainty_ece = calculate_calibration_metrics(uncertainty_u_logs, filtered_grasp)
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Baseline**")
                                    if baseline_ece is not None:
                                        st.metric("ECE", f"{baseline_ece:.3f}")
                                    else:
                                        st.info("Insufficient data")
                                
                                with col2:
                                    st.markdown("**Uncertainty-Aware**")
                                    if uncertainty_ece is not None:
                                        delta = (uncertainty_ece - baseline_ece) if baseline_ece else None
                                        st.metric("ECE", f"{uncertainty_ece:.3f}", 
                                                 delta=f"{delta:.3f}" if delta else None,
                                                 delta_color="inverse")  # Lower is better
                                    else:
                                        st.info("Insufficient data")
                        
                        else:
                            st.warning(f"⚠️ Insufficient calibration data ({len(calibration_data) if calibration_data else 0} bins)")
                            st.info("💡 **Recommendation:** Run at least 20 experiments with varying outcomes for reliable calibration analysis.")
                
                # ========== TAB 3: STATISTICAL TESTS ==========
                with analysis_tabs[2]:
                    st.markdown("### 📊 Statistical Significance Tests")
                    st.markdown("Determine if observed differences are statistically significant or due to random chance.")
                    
                    if not filtered_grasp or len(filtered_grasp) < 10:
                        st.warning("⚠️ Insufficient data for statistical tests. Need at least 10 experiments.")
                        st.info("💡 Run batch experiments:\n```bash\npython notebooks/batch_experiments.py --mode baseline_vs_uncertainty\n```")
                    else:
                        # Import scipy for statistical tests
                        try:
                            from scipy import stats
                            from scipy.stats import mannwhitneyu, chi2_contingency
                        except ImportError:
                            st.error("❌ scipy not installed. Run: `pip install scipy`")
                            st.stop()
                        
                        # Test selection
                        test_type = st.selectbox(
                            "Select Statistical Test",
                            [
                                "Success Rate Comparison (Chi-square)",
                                "Calibration Quality (t-test)",
                                "Model Performance (Mann-Whitney U)",
                                "Multiple Prompt Variants (ANOVA)"
                            ],
                            help="Choose the appropriate test for your research question"
                        )
                        
                        st.markdown("---")
                        
                        # ========== CHI-SQUARE TEST ==========
                        if test_type == "Success Rate Comparison (Chi-square)":
                            st.markdown("#### χ² Test: Success Rate Comparison")
                            st.markdown("**Research Question:** Do baseline and uncertainty-aware prompts have significantly different success rates?")
                            
                            baseline_grasps = [g for g in filtered_grasp if g.get("experiment_group") == "baseline"]
                            uncertainty_grasps = [g for g in filtered_grasp if g.get("experiment_group") == "uncertainty_aware"]
                            
                            if not baseline_grasps or not uncertainty_grasps:
                                st.warning("Need both baseline and uncertainty-aware experiments")
                            else:
                                # Create contingency table
                                baseline_success = sum(1 for g in baseline_grasps if g.get("success"))
                                baseline_fail = len(baseline_grasps) - baseline_success
                                
                                uncertainty_success = sum(1 for g in uncertainty_grasps if g.get("success"))
                                uncertainty_fail = len(uncertainty_grasps) - uncertainty_success
                                
                                contingency = np.array([
                                    [baseline_success, baseline_fail],
                                    [uncertainty_success, uncertainty_fail]
                                ])
                                
                                # Perform chi-square test
                                chi2, p_value, dof, expected = chi2_contingency(contingency)
                                
                                # Display results
                                col1, col2, col3, col4 = st.columns(4)
                                
                                col1.metric("χ² Statistic", f"{chi2:.3f}")
                                col2.metric("p-value", f"{p_value:.4f}")
                                col3.metric("Degrees of Freedom", dof)
                                
                                if p_value < 0.05:
                                    col4.metric("Result", "✅ Significant", help="p < 0.05: Reject null hypothesis")
                                    st.success(f"**Conclusion:** The difference in success rates is statistically significant (p = {p_value:.4f})")
                                elif p_value < 0.10:
                                    col4.metric("Result", "⚠️ Marginal", help="0.05 < p < 0.10: Marginally significant")
                                    st.warning(f"**Conclusion:** Marginally significant difference (p = {p_value:.4f})")
                                else:
                                    col4.metric("Result", "❌ Not Significant", help="p ≥ 0.05: Fail to reject null hypothesis")
                                    st.info(f"**Conclusion:** No significant difference found (p = {p_value:.4f})")
                                
                                # Contingency table visualization
                                st.markdown("**Contingency Table:**")
                                contingency_df = pd.DataFrame(
                                    contingency,
                                    columns=['Success', 'Failure'],
                                    index=['Baseline', 'Uncertainty-Aware']
                                )
                                st.dataframe(contingency_df, use_container_width=True)
                                
                                # Effect size (Cramér's V)
                                n = contingency.sum()
                                cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                                
                                st.markdown(f"**Effect Size (Cramér's V):** {cramers_v:.3f}")
                                if cramers_v < 0.1:
                                    st.text("→ Small effect")
                                elif cramers_v < 0.3:
                                    st.text("→ Medium effect")
                                else:
                                    st.text("→ Large effect")
                        
                        # ========== T-TEST FOR CALIBRATION ==========
                        elif test_type == "Calibration Quality (t-test)":
                            st.markdown("#### t-test: Calibration Quality Comparison")
                            st.markdown("**Research Question:** Does uncertainty-aware prompting produce better-calibrated models?")
                            
                            baseline_u_logs = [log for log in filtered_uncertainty if log.get("experiment_group") == "baseline"]
                            uncertainty_u_logs = [log for log in filtered_uncertainty if log.get("experiment_group") == "uncertainty_aware"]
                            
                            if not baseline_u_logs or not uncertainty_u_logs:
                                st.warning("Need both baseline and uncertainty-aware experiments")
                            else:
                                # Calculate ECE for each experiment
                                baseline_eces = []
                                for log in baseline_u_logs:
                                    matching_grasps = [g for g in filtered_grasp if g.get("experiment_id") == log.get("experiment_id")]
                                    if matching_grasps:
                                        _, ece = calculate_calibration_metrics([log], matching_grasps)
                                        if ece is not None:
                                            baseline_eces.append(ece)
                                
                                uncertainty_eces = []
                                for log in uncertainty_u_logs:
                                    matching_grasps = [g for g in filtered_grasp if g.get("experiment_id") == log.get("experiment_id")]
                                    if matching_grasps:
                                        _, ece = calculate_calibration_metrics([log], matching_grasps)
                                        if ece is not None:
                                            uncertainty_eces.append(ece)
                                
                                if len(baseline_eces) < 2 or len(uncertainty_eces) < 2:
                                    st.warning("Need at least 2 ECE values per group for t-test")
                                else:
                                    # Perform t-test
                                    t_stat, p_value = stats.ttest_ind(baseline_eces, uncertainty_eces)
                                    
                                    # Display results
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    col1.metric("t-statistic", f"{t_stat:.3f}")
                                    col2.metric("p-value", f"{p_value:.4f}")
                                    col3.metric("df", len(baseline_eces) + len(uncertainty_eces) - 2)
                                    
                                    if p_value < 0.05:
                                        col4.metric("Result", "✅ Significant")
                                        st.success(f"**Conclusion:** Calibration quality differs significantly (p = {p_value:.4f})")
                                    else:
                                        col4.metric("Result", "❌ Not Significant")
                                        st.info(f"**Conclusion:** No significant difference (p = {p_value:.4f})")
                                    
                                    # Effect size (Cohen's d)
                                    mean_baseline = np.mean(baseline_eces)
                                    mean_uncertainty = np.mean(uncertainty_eces)
                                    pooled_std = np.sqrt((np.var(baseline_eces) + np.var(uncertainty_eces)) / 2)
                                    cohens_d = (mean_uncertainty - mean_baseline) / pooled_std if pooled_std > 0 else 0
                                    
                                    st.markdown(f"**Effect Size (Cohen's d):** {cohens_d:.3f}")
                                    if abs(cohens_d) < 0.2:
                                        st.text("→ Negligible effect")
                                    elif abs(cohens_d) < 0.5:
                                        st.text("→ Small effect")
                                    elif abs(cohens_d) < 0.8:
                                        st.text("→ Medium effect")
                                    else:
                                        st.text("→ Large effect")
                                    
                                    # Box plot comparison
                                    st.markdown("**Distribution Comparison:**")
                                    comparison_df = pd.DataFrame({
                                        'ECE': baseline_eces + uncertainty_eces,
                                        'Group': ['Baseline']*len(baseline_eces) + ['Uncertainty-Aware']*len(uncertainty_eces)
                                    })
                                    
                                    box_plot = alt.Chart(comparison_df).mark_boxplot(size=60).encode(
                                    x=alt.X('Group:N', title=None),
                                    y=alt.Y('ECE:Q', title='Expected Calibration Error', scale=alt.Scale(zero=False)),
                                    color=alt.Color('Group:N', legend=None)
                                    ).properties(height=300)
                                    st.altair_chart(box_plot, use_container_width=True)
                    
                        # ========== MANN-WHITNEY U TEST ==========
                        elif test_type == "Model Performance (Mann-Whitney U)":
                            st.markdown("#### Mann-Whitney U Test: Non-parametric Model Comparison")
                            st.markdown("**Research Question (RQ4):** Do SVLMs perform comparably to large VLMs?")
                            
                            # Extract model names from logs
                            model_names = set()
                            for log in filtered_uncertainty:
                                models = log.get("model", {})
                                for module, model_info in models.items():
                                    if isinstance(model_info, dict):
                                        model_names.add(model_info.get("model_name"))
                            
                            if len(model_names) < 2:
                                st.warning("Need experiments with at least 2 different models")
                                st.info(f"Found models: {', '.join(model_names) if model_names else 'None'}")
                            else:
                                model_list = sorted(model_names)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    model_a = st.selectbox("Model A", options=model_list, index=0)
                                with col2:
                                    model_b = st.selectbox("Model B", options=model_list, index=min(1, len(model_list)-1))
                                
                                if model_a == model_b:
                                    st.warning("Please select two different models")
                                else:
                                    # Get success rates for each model
                                    def get_model_success_rates(model_name):
                                        exp_ids_with_model = []
                                        for log in filtered_uncertainty:
                                            models = log.get("model", {})
                                            for module, model_info in models.items():
                                                if isinstance(model_info, dict) and model_info.get("model_name") == model_name:
                                                    exp_ids_with_model.append(log.get("experiment_id"))
                                                    break
                                        
                                        success_rates = []
                                        for exp_id in set(exp_ids_with_model):
                                            exp_grasps = [g for g in filtered_grasp if g.get("experiment_id") == exp_id]
                                            if exp_grasps:
                                                success_rate = sum(1 for g in exp_grasps if g.get("success")) / len(exp_grasps)
                                                success_rates.append(success_rate)
                                        return success_rates
                                    
                                    rates_a = get_model_success_rates(model_a)
                                    rates_b = get_model_success_rates(model_b)
                                    
                                    if len(rates_a) < 2 or len(rates_b) < 2:
                                        st.warning(f"Need at least 2 experiments per model (found: {len(rates_a)} for {model_a}, {len(rates_b)} for {model_b})")
                                    else:
                                        # Perform Mann-Whitney U test
                                        u_stat, p_value = mannwhitneyu(rates_a, rates_b, alternative='two-sided')
                                        
                                        # Display results
                                        col1, col2, col3 = st.columns(3)
                                        
                                        col1.metric("U Statistic", f"{u_stat:.1f}")
                                        col2.metric("p-value", f"{p_value:.4f}")
                                        
                                        if p_value < 0.05:
                                            col3.metric("Result", "✅ Significant")
                                            st.success(f"**Conclusion:** Models show significantly different performance (p = {p_value:.4f})")
                                        else:
                                            col3.metric("Result", "❌ Not Significant")
                                            st.info(f"**Conclusion:** No significant difference found (p = {p_value:.4f})")
                                            st.success("💡 **Research Implication:** SVLMs may be viable alternatives!")
                                        
                                        # Descriptive statistics
                                        st.markdown("**Descriptive Statistics:**")
                                        stats_df = pd.DataFrame({
                                            'Model': [model_a, model_b],
                                            'N': [len(rates_a), len(rates_b)],
                                            'Mean Success': [f"{np.mean(rates_a):.3f}", f"{np.mean(rates_b):.3f}"],
                                            'Median': [f"{np.median(rates_a):.3f}", f"{np.median(rates_b):.3f}"],
                                            'Std Dev': [f"{np.std(rates_a):.3f}", f"{np.std(rates_b):.3f}"]
                                        })
                                        st.dataframe(stats_df, use_container_width=True)
                                        
                                        # Distribution comparison
                                        st.markdown("**Distribution Comparison:**")
                                        comparison_df = pd.DataFrame({
                                            'Success Rate': rates_a + rates_b,
                                            'Model': [model_a]*len(rates_a) + [model_b]*len(rates_b)
                                        })
                                        
                                        violin_plot = alt.Chart(comparison_df).transform_density(
                                            'Success Rate',
                                            as_=['Success Rate', 'density'],
                                            groupby=['Model']
                                        ).mark_area(orient='horizontal', opacity=0.5).encode(
                                            y=alt.Y('Success Rate:Q', scale=alt.Scale(domain=[0, 1])),
                                            x=alt.X('density:Q', title=None, axis=alt.Axis(labels=False, ticks=False)),
                                            color=alt.Color('Model:N'),
                                            row=alt.Row('Model:N', title=None)
                                        ).properties(height=150)
                                        
                                        st.altair_chart(violin_plot, use_container_width=True)
                        
                        # ========== ANOVA TEST ==========
                        elif test_type == "Multiple Prompt Variants (ANOVA)":
                            st.markdown("#### ANOVA: Compare Multiple Prompt Variants")
                            st.markdown("**Research Question (RQ1):** Which prompt engineering strategy is most effective?")
                            
                            # Get unique prompt types
                            prompt_types = set(log.get("experiment_group") for log in filtered_uncertainty)
                            
                            # Also check for different prompt names if available
                            prompt_names = set()
                            for log in filtered_uncertainty:
                                pnames = log.get("prompt_name", {})
                                if isinstance(pnames, dict):
                                    for module, pname in pnames.items():
                                        if pname:
                                            # Extract prompt variant (e.g., "confidence", "hedging")
                                            parts = pname.split('_')
                                            if len(parts) > 2:
                                                prompt_names.add(parts[-1])  # Last part is usually the variant
                            
                            if len(prompt_types) < 3 and len(prompt_names) < 3:
                                st.warning("ANOVA requires at least 3 groups. Run more prompt variants.")
                                st.info(f"Currently have: {len(prompt_types)} experiment groups, {len(prompt_names)} prompt variants")
                            else:
                                metric_choice = st.radio(
                                    "Select metric to compare",
                                    ["Success Rate", "Calibration Quality (ECE)", "Average Retries"],
                                    horizontal=True
                                )
                                
                                # Group data by prompt variant
                                groups = {}
                                
                                for log in filtered_uncertainty:
                                    # Determine group identifier
                                    group_id = log.get("experiment_group", "unknown")
                                    
                                    exp_id = log.get("experiment_id")
                                    exp_grasps = [g for g in filtered_grasp if g.get("experiment_id") == exp_id]
                                    
                                    if exp_grasps:
                                        if group_id not in groups:
                                            groups[group_id] = []
                                        
                                        if metric_choice == "Success Rate":
                                            value = sum(1 for g in exp_grasps if g.get("success")) / len(exp_grasps)
                                        elif metric_choice == "Calibration Quality (ECE)":
                                            _, ece = calculate_calibration_metrics([log], exp_grasps)
                                            value = ece if ece is not None else None
                                        else:  # Average Retries
                                            value = np.mean([g.get("retries", 0) for g in exp_grasps])
                                        
                                        if value is not None:
                                            groups[group_id].append(value)
                                
                                # Filter groups with sufficient data
                                valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}
                                
                                if len(valid_groups) < 3:
                                    st.warning(f"Need at least 3 groups with 2+ samples each. Have: {len(valid_groups)}")
                                else:
                                    # Perform ANOVA
                                    f_stat, p_value = stats.f_oneway(*valid_groups.values())
                                    
                                    # Display results
                                    col1, col2, col3 = st.columns(3)
                                    
                                    col1.metric("F-statistic", f"{f_stat:.3f}")
                                    col2.metric("p-value", f"{p_value:.4f}")
                                    col3.metric("Groups", len(valid_groups))
                                    
                                    if p_value < 0.05:
                                        st.success(f"**Conclusion:** Significant difference exists among prompt variants (p = {p_value:.4f})")
                                        st.info("💡 Consider post-hoc tests (e.g., Tukey HSD) to identify which specific pairs differ")
                                    else:
                                        st.info(f"**Conclusion:** No significant differences found (p = {p_value:.4f})")
                                    
                                    # Group statistics table
                                    st.markdown("**Group Statistics:**")
                                    stats_data = []
                                    for group_name, values in valid_groups.items():
                                        stats_data.append({
                                            'Group': group_name,
                                            'N': len(values),
                                            'Mean': f"{np.mean(values):.3f}",
                                            'Std Dev': f"{np.std(values):.3f}",
                                            'Min': f"{np.min(values):.3f}",
                                            'Max': f"{np.max(values):.3f}"
                                        })
                                    
                                    stats_df = pd.DataFrame(stats_data)
                                    st.dataframe(stats_df, use_container_width=True)
                                    
                                    # Box plot comparison
                                    st.markdown("**Distribution Comparison:**")
                                    plot_data = []
                                    for group_name, values in valid_groups.items():
                                        for value in values:
                                            plot_data.append({'Group': group_name, 'Value': value})
                                    
                                    plot_df = pd.DataFrame(plot_data)
                                    
                                    box_plot = alt.Chart(plot_df).mark_boxplot(size=40).encode(
                                        x=alt.X('Group:N', title=None, axis=alt.Axis(labelAngle=-45)),
                                        y=alt.Y('Value:Q', title=metric_choice),
                                        color=alt.Color('Group:N', legend=None)
                                    ).properties(height=350)
                                    
                                    st.altair_chart(box_plot, use_container_width=True)
                
                # ========== TAB 4: LEADERBOARD ==========
                with analysis_tabs[3]:
                    st.markdown("### 🏆 Model Configuration Leaderboard")
                    st.markdown("Ranked by composite performance score (calibration + success rate)")
                    
                    if not filtered_grasp:
                        st.info("No grasp data available for leaderboard")
                    else:
                        # Build leaderboard data
                        leaderboard_data = []
                        
                        for log in filtered_uncertainty:
                            exp_id = log.get("experiment_id")
                            exp_group = log.get("experiment_group", "unknown")
                            
                            # Get model info
                            models = log.get("model", {})
                            model_names = []
                            for module in ["grounder", "planner", "ranker"]:
                                if module in models and isinstance(models[module], dict):
                                    model_names.append(models[module].get("model_name", "unknown"))
                            
                            model_str = " / ".join(model_names) if model_names else "unknown"
                            
                            # Get performance metrics
                            exp_grasps = [g for g in filtered_grasp if g.get("experiment_id") == exp_id]
                            
                            if exp_grasps:
                                success_rate = sum(1 for g in exp_grasps if g.get("success")) / len(exp_grasps)
                                avg_retries = np.mean([g.get("retries", 0) for g in exp_grasps])
                                
                                # Get calibration
                                _, ece = calculate_calibration_metrics([log], exp_grasps)
                                
                                # Calculate composite score
                                # Normalize: success_rate (higher better), 1-ECE (higher better), 1/(1+retries) (higher better)
                                norm_success = success_rate
                                norm_calib = (1 - ece) if ece is not None else 0.5
                                norm_retries = 1 / (1 + avg_retries)
                                
                                composite = (norm_success * 0.5 + norm_calib * 0.3 + norm_retries * 0.2)
                                
                                leaderboard_data.append({
                                    'Experiment ID': exp_id[:16] + "...",
                                    'Group': exp_group,
                                    'Model': model_str,
                                    'Success Rate': success_rate,
                                    'ECE': ece if ece is not None else np.nan,
                                    'Avg Retries': avg_retries,
                                    'Composite Score': composite
                                })
                        
                        if leaderboard_data:
                            lb_df = pd.DataFrame(leaderboard_data)
                            lb_df = lb_df.sort_values('Composite Score', ascending=False)
                            
                            # Format for display
                            display_df = lb_df.copy()
                            display_df['Success Rate'] = display_df['Success Rate'].apply(lambda x: f"{x:.1%}")
                            display_df['ECE'] = display_df['ECE'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
                            display_df['Avg Retries'] = display_df['Avg Retries'].apply(lambda x: f"{x:.2f}")
                            display_df['Composite Score'] = display_df['Composite Score'].apply(lambda x: f"{x:.3f}")
                            
                            # Highlight top 3
                            def highlight_top3(row):
                                if row.name == 0:
                                    return ['background-color: #FFD700'] * len(row)  # Gold
                                elif row.name == 1:
                                    return ['background-color: #C0C0C0'] * len(row)  # Silver
                                elif row.name == 2:
                                    return ['background-color: #CD7F32'] * len(row)  # Bronze
                                return [''] * len(row)
                            
                            styled_df = display_df.style.apply(highlight_top3, axis=1)
                            
                            st.dataframe(styled_df, use_container_width=True, height=400)
                            
                            # Top performer summary
                            if len(lb_df) > 0:
                                st.markdown("---")
                                st.markdown("#### 🥇 Top Performer")
                                
                                top = lb_df.iloc[0]
                                col1, col2, col3, col4 = st.columns(4)
                                
                                col1.metric("Experiment", top['Experiment ID'])
                                col2.metric("Success Rate", f"{top['Success Rate']:.1%}")
                                col3.metric("ECE", f"{top['ECE']:.3f}" if not pd.isna(top['ECE']) else "N/A")
                                col4.metric("Score", f"{top['Composite Score']:.3f}")
                        else:
                            st.info("No leaderboard data available")
                
                # ========== TAB 5: RAW DATA ==========
                with analysis_tabs[4]:
                    st.markdown("### 📋 Raw Uncertainty Logs")
                    
                    # Display options
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        show_metadata = st.checkbox("Show metadata columns", value=False)
                    with col2:
                        rows_to_show = st.number_input("Rows", min_value=10, max_value=1000, value=100, step=10)
                    
                    # Prepare dataframe
                    df_display = pd.DataFrame(filtered_uncertainty)
                    
                    if not show_metadata:
                        # Hide complex nested columns
                        simple_cols = [col for col in df_display.columns if not col.startswith('metadata')]
                        df_display = df_display[simple_cols]
                    
                    st.dataframe(df_display.head(rows_to_show), use_container_width=True, height=400)
                    
                    # Download options
                    st.markdown("---")
                    st.markdown("### 💾 Export Data")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download uncertainty logs
                        json_str = json.dumps(filtered_uncertainty, indent=2)
                        st.download_button(
                            "📥 Download Uncertainty Logs (JSON)",
                            data=json_str,
                            file_name=f"uncertainty_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    with col2:
                        # Download grasp logs if available
                        if filtered_grasp:
                            grasp_json = json.dumps(filtered_grasp, indent=2)
                            st.download_button(
                                "📥 Download Grasp Logs (JSON)",
                                data=grasp_json,
                                file_name=f"grasp_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )

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