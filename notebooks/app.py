# app.py
import shlex
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

with tabs[3]:
    st.header("🧠 Uncertainty Analysis")

    if not os.path.exists(LOG_PATH_UNCERT):
        st.info("No uncertainty log found. Run experiments to generate one.")
    else:
        with open(LOG_PATH_UNCERT, "r") as f:
            logs = [json.loads(line) for line in f if line.strip()]
        df_uncert = pd.json_normalize(logs, sep="_")

        st.write("Detected columns:", list(df_uncert.columns))

        # ---- Helper function to extract entropy & confidence ----
        def extract_values(metadata):
            """Extract mean entropy and confidence values from nested lists or dicts."""
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
            elif isinstance(metadata, list):
                entropies = [x.get("entropy") for x in metadata if isinstance(x, dict) and "entropy" in x]
                confidences = [
                    (x.get("confidence") if x.get("confidence") != -1 else np.nan)
                    for x in metadata if isinstance(x, dict) and "confidence" in x
                ]
                if entropies:
                    out["entropy"] = np.mean(entropies)
                if confidences:
                    out["confidence"] = np.nanmean(confidences)
            return out

        # ---- Extract values from all metadata_* columns ----
        metadata_cols = [c for c in df_uncert.columns if c.startswith("metadata_")]
        all_rows = []

        for i, row in df_uncert.iterrows():
            merged = {"timestamp": row["timestamp"]}
            for col in metadata_cols:
                values = extract_values(row[col])
                for k, v in values.items():
                    merged[f"{col.replace('metadata_', '')}_{k}"] = v
            all_rows.append(merged)

        df_unc = pd.DataFrame(all_rows)

        # ---- Merge model names ----
        for model_type in ["ranker", "planner", "grounder"]:
            model_col = f"model_{model_type}_model_name"
            if model_col in df_uncert.columns:
                df_unc[f"{model_type}_model"] = df_uncert[model_col]
            else:
                df_unc[f"{model_type}_model"] = "unknown"

        # ---- Data Quality Checks ----
        st.subheader("🔍 Data Quality Report")
        col_quality1, col_quality2 = st.columns(2)
        
        with col_quality1:
            st.markdown("**Missing Values**")
            missing_data = df_unc.isnull().sum()
            if missing_data.any() and missing_data.sum() > 0:
                st.dataframe(missing_data[missing_data > 0].to_frame("Count"), height=150)
            else:
                st.success("✓ No missing values")
        
        with col_quality2:
            st.markdown("**Data Summary**")
            st.metric("Total Records", len(df_unc))
            st.metric("Time Span", f"{(pd.to_datetime(df_unc['timestamp']).max() - pd.to_datetime(df_unc['timestamp']).min()).days} days")

        # ---- Display data ----
        if not df_unc.empty:
            st.subheader("📊 Extracted Entropy & Confidence Values")
            st.dataframe(df_unc, width='stretch', height=400)

            # ---- Split columns ----
            entropy_cols = [c for c in df_unc.columns if c.endswith("_entropy")]
            conf_cols = [c for c in df_unc.columns if c.endswith("_confidence")]

            # Check for constant values
            constant_cols = []
            for col in entropy_cols + conf_cols:
                if df_unc[col].notna().sum() > 0 and df_unc[col].nunique() == 1:
                    constant_cols.append(col)
            
            if constant_cols:
                st.warning(f"⚠️ Constant value columns detected: {', '.join(constant_cols)}")

            # ---- Melt for plotting ----
            def melt_df(cols, metric_name):
                return df_unc.melt(
                    id_vars=["timestamp", "ranker_model", "planner_model", "grounder_model"],
                    value_vars=cols,
                    var_name="metric",
                    value_name="value"
                ).assign(type=metric_name)

            melted_entropy = melt_df(entropy_cols, "entropy")
            melted_conf = melt_df(conf_cols, "confidence")

            # ---- Charts ----
            st.subheader("📈 Evolution Over Time")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔵 Entropy Trends")
                if not melted_entropy.empty and melted_entropy['value'].notna().any():
                    chart_entropy = (
                        alt.Chart(melted_entropy.dropna(subset=['value']))
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("timestamp:T", title="Time"),
                            y=alt.Y("value:Q", title="Entropy"),
                            color=alt.Color("metric:N", legend=alt.Legend(orient="bottom")),
                            tooltip=["timestamp", "metric", "value", "ranker_model", "planner_model", "grounder_model"]
                        )
                        .properties(height=300)
                        .interactive()
                    )
                    st.altair_chart(chart_entropy, width='stretch')
                else:
                    st.info("No entropy data available")

            with col2:
                st.markdown("### 🟢 Confidence Trends")
                if not melted_conf.empty and melted_conf['value'].notna().any():
                    chart_conf = (
                        alt.Chart(melted_conf.dropna(subset=['value']))
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("timestamp:T", title="Time"),
                            y=alt.Y("value:Q", title="Confidence"),
                            color=alt.Color("metric:N", legend=alt.Legend(orient="bottom")),
                            tooltip=["timestamp", "metric", "value", "ranker_model", "planner_model", "grounder_model"]
                        )
                        .properties(height=300)
                        .interactive()
                    )
                    st.altair_chart(chart_conf, width='stretch')
                else:
                    st.info("No confidence data available")

            # ---- Summary Stats ----
            st.subheader("📋 Summary Statistics")

            melted_all = pd.concat([melted_entropy, melted_conf], ignore_index=True)
            summary = (
                melted_all.groupby(["type", "metric"])["value"]
                .agg(["count", "mean", "std", "min", "max"])
                .reset_index()
                .round(5)
            )
            st.dataframe(summary, width='stretch')

            # ---- Correlation Analysis ----
            st.subheader("🔗 Entropy-Confidence Correlation")
            
            corr_cols = [c for c in df_unc.columns if '_entropy' in c or '_confidence' in c]
            if len(corr_cols) >= 2:
                corr_data = df_unc[corr_cols].dropna()
                if len(corr_data) > 1:
                    corr_matrix = corr_data.corr()
                    
                    # Create heatmap using altair
                    corr_reset = corr_matrix.reset_index().melt(id_vars='index')
                    corr_reset.columns = ['Variable 1', 'Variable 2', 'Correlation']
                    
                    heatmap = alt.Chart(corr_reset).mark_rect().encode(
                        x=alt.X('Variable 1:N', title=None),
                        y=alt.Y('Variable 2:N', title=None),
                        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
                        tooltip=['Variable 1', 'Variable 2', alt.Tooltip('Correlation:Q', format='.3f')]
                    ).properties(
                        width=600,
                        height=600
                    )
                    
                    st.altair_chart(heatmap, width='stretch')
                else:
                    st.info("Not enough data points for correlation analysis")
            else:
                st.info("Need at least 2 metrics for correlation analysis")

            # ---- Anomaly Detection ----
            st.subheader("⚠️ Anomaly Detection (Z-score > 2)")
            
            anomalies_found = False
            for col in entropy_cols + conf_cols:
                col_data = df_unc[col].dropna()
                if len(col_data) > 2 and col_data.std() > 0:
                    z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                    outlier_indices = z_scores[z_scores > 2].index
                    
                    if len(outlier_indices) > 0:
                        anomalies_found = True
                        with st.expander(f"🔴 {col}: {len(outlier_indices)} anomalies found"):
                            outlier_data = df_unc.loc[outlier_indices, ["timestamp", col] + 
                                                       ["ranker_model", "planner_model", "grounder_model"]]
                            st.dataframe(outlier_data, width='stretch')
            
            if not anomalies_found:
                st.success("✓ No significant anomalies detected")

            # ---- Per-Model Summary (FIXED) ----
            st.subheader("🤖 Mean Metric Values per Model")

            # FIX: Add "type" and "metric" to id_vars
            model_melted = melted_all.melt(
                id_vars=["timestamp", "value", "type", "metric"],
                value_vars=["ranker_model", "planner_model", "grounder_model"],
                var_name="model_type",
                value_name="model_name",
            )
            
            model_summary = (
                model_melted.groupby(["model_type", "model_name", "type"])["value"]
                .agg(["count", "mean", "std", "min", "max"])
                .reset_index()
                .round(5)
            )
            st.dataframe(model_summary, width='stretch')

            # ---- Model Configuration Leaderboard ----
            st.subheader("🏆 Model Configuration Performance")
            
            if len(df_unc) > 0 and (entropy_cols or conf_cols):
                config_data = []
                
                for _, row in df_unc.iterrows():
                    config = {
                        'ranker': row.get('ranker_model', 'unknown'),
                        'planner': row.get('planner_model', 'unknown'),
                        'grounder': row.get('grounder_model', 'unknown')
                    }
                    
                    # Calculate average entropy (lower is better)
                    avg_entropy = row[[c for c in entropy_cols]].mean() if entropy_cols else np.nan
                    
                    # Calculate average confidence (higher is better)
                    avg_conf = row[[c for c in conf_cols]].mean() if conf_cols else np.nan
                    
                    config_data.append({
                        'ranker_model': config['ranker'],
                        'planner_model': config['planner'],
                        'grounder_model': config['grounder'],
                        'avg_entropy': avg_entropy,
                        'avg_confidence': avg_conf
                    })
                
                config_df = pd.DataFrame(config_data)
                
                # Group by configuration
                config_summary = config_df.groupby(['ranker_model', 'planner_model', 'grounder_model']).agg({
                    'avg_entropy': ['mean', 'std', 'count'],
                    'avg_confidence': ['mean', 'std']
                }).reset_index()
                
                config_summary.columns = ['_'.join(col).strip('_') for col in config_summary.columns.values]
                
                # Calculate composite score (normalize metrics)
                if 'avg_entropy_mean' in config_summary.columns and 'avg_confidence_mean' in config_summary.columns:
                    # Normalize to 0-1 range
                    if config_summary['avg_entropy_mean'].std() > 0:
                        norm_entropy = (config_summary['avg_entropy_mean'].max() - config_summary['avg_entropy_mean']) / \
                                      (config_summary['avg_entropy_mean'].max() - config_summary['avg_entropy_mean'].min())
                    else:
                        norm_entropy = 0
                    
                    if config_summary['avg_confidence_mean'].std() > 0:
                        norm_conf = (config_summary['avg_confidence_mean'] - config_summary['avg_confidence_mean'].min()) / \
                                   (config_summary['avg_confidence_mean'].max() - config_summary['avg_confidence_mean'].min())
                    else:
                        norm_conf = 0
                    
                    config_summary['composite_score'] = (norm_entropy + norm_conf) / 2
                    config_summary = config_summary.sort_values('composite_score', ascending=False)
                
                st.dataframe(config_summary.round(4), width='stretch')
                
                # Show top configuration
                if len(config_summary) > 0 and 'composite_score' in config_summary.columns:
                    best_config = config_summary.iloc[0]
                    st.success(f"🥇 **Best Configuration:** "
                             f"Ranker={best_config['ranker_model']}, "
                             f"Planner={best_config['planner_model']}, "
                             f"Grounder={best_config['grounder_model']} "
                             f"(Score: {best_config['composite_score']:.3f})")

            # ---- Statistical Significance Tests ----
            st.subheader("📊 Statistical Significance Tests (ANOVA)")
            
            if len(df_unc) >= 10:  # Need enough samples
                from scipy import stats
                
                sig_results = []
                
                for metric in entropy_cols + conf_cols:
                    for model_type in ['ranker_model', 'planner_model', 'grounder_model']:
                        groups = [df_unc[df_unc[model_type] == m][metric].dropna() 
                                 for m in df_unc[model_type].unique()]
                        groups = [g for g in groups if len(g) > 0]
                        
                        if len(groups) >= 2:
                            try:
                                f_stat, p_value = stats.f_oneway(*groups)
                                sig_results.append({
                                    'Metric': metric,
                                    'Model Type': model_type.replace('_model', ''),
                                    'F-statistic': f_stat,
                                    'p-value': p_value,
                                    'Significant': '✓' if p_value < 0.05 else '✗'
                                })
                            except:
                                pass
                
                if sig_results:
                    sig_df = pd.DataFrame(sig_results)
                    st.dataframe(sig_df.round(4), width='stretch')
                    
                    significant_count = (sig_df['p-value'] < 0.05).sum()
                    if significant_count > 0:
                        st.info(f"ℹ️ Found {significant_count} statistically significant differences (p < 0.05)")
                else:
                    st.info("Unable to perform statistical tests with current data")
            else:
                st.info("Need at least 10 records for statistical significance testing")

        else:
            st.info("No entropy/confidence values found in metadata.")

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
    litellm_col1, litellm_col2, litellm_col3 = st.columns([2, 2, 1])
    
    with litellm_col1:
        if st.button("🔍 Check LiteLLM Status", use_container_width=True):
            with st.spinner("Checking LiteLLM server..."):
                try:
                    # Try multiple endpoints
                    endpoints = [
                        "http://localhost:4000/health",
                        "http://localhost:4000/",
                        "http://127.0.0.1:4000/health",
                        "http://0.0.0.0:4000/health",
                        "http://0.0.0.0:4000/"
                    ]
                    
                    litellm_found = False
                    for endpoint in endpoints:
                        try:
                            response = requests.get(endpoint, timeout=3)
                            if response.status_code in [200, 404]:  # 404 means server is up but no route
                                st.session_state['litellm_running'] = True
                                litellm_found = True
                                st.success(f"✅ LiteLLM is running on port 4000")
                                break
                        except:
                            continue
                    
                    if litellm_found:
                        # Get available models
                        try:
                            models_response = requests.get("http://localhost:4000/v1/models", timeout=3)
                            if models_response.status_code == 200:
                                models_data = models_response.json()
                                st.session_state['available_models'] = [m['id'] for m in models_data.get('data', [])]
                                st.info(f"Found {len(st.session_state['available_models'])} models")
                        except Exception as e:
                            st.warning(f"Models endpoint not available, using defaults. Error: {e}")
                            st.session_state['available_models'] = ['gpt-4o', 'gpt-4o-mini', 'claude-3-5-sonnet-20241022']
                    else:
                        st.session_state['litellm_running'] = False
                        st.error("❌ LiteLLM not running on port 4000")
                        
                except Exception as e:
                    st.session_state['litellm_running'] = False
                    st.error(f"❌ Error checking LiteLLM: {e}")
    
    with litellm_col2:
        if st.button("▶️ Start LiteLLM", use_container_width=True):
            with st.spinner("Starting LiteLLM server..."):
                try:
                    import subprocess
                    litellm_config_path = "config/litellm/config.yaml"
                    if os.path.exists(litellm_config_path):
                        subprocess.Popen(["litellm", "--config", litellm_config_path], 
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL)
                        time.sleep(3)
                        st.success("✅ LiteLLM started (check status)")
                        st.rerun()
                    else:
                        st.error(f"❌ Config file not found: {litellm_config_path}")
                except Exception as e:
                    st.error(f"❌ Failed to start LiteLLM: {e}")
    
    with litellm_col3:
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
                    
                    prompt_name = st.selectbox(
                        "Prompt File",
                        options=all_prompts,
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
        'request': {'model_name': 'gpt-4o', 'temperature': 0.0, 'n': 2, 'max_tokens': 256, 'logprobs': False}
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