# **OWG Experiment Dashboard**

A Streamlit-based interface for configuring, monitoring, and executing **Open-World Grasping (OWG)** experiments.
The dashboard integrates LiteLLM logs, uncertainty analysis, prompt engineering, configurable pipelines, and experiment execution.

---

## **🌐 Features Overview**

### **1. 🔍 Experiment Logs**

View and analyze LiteLLM API activity:

* Load logs from `logs/litellm_logs.jsonl`
* Filter by model, status, and user
* Display costs, responses, timestamps
* Visual charts:

  * Cost over time
  * Request counts by status
* Summary statistics (success rate, total cost)

---

### **2. 🧩 Prompt Engineering**

A prompt management system for OWG tasks:

* Browse built-in prompts (`prompts/uncertainty_aware`)
* Browse user-created prompts (`prompts/user_defined`)
* Create, edit, and save new prompts
* Delete user-defined prompts
* Preview prompt content

---

### **3. 📈 Metrics Overview**

Monitor outcomes from OWG grasp attempts:

* Load logs from `logs/experiment_metrics.jsonl`
* Display grasp attempt details:

  * Success / failure
  * Object ID
  * Position
  * Retries
* Compute:

  * Total grasps
  * Success rate
  * Avg retries
* Visualizations:

  * Success rate per object
  * Timeline of grasp attempts

---

### **4. 🧠 Uncertainty Analysis**

Analyze uncertainty-aware metadata (entropy & confidence):

* Load logs from `logs/uncertainty_logs.jsonl`
* Automatic extraction of nested metadata
* Reports on missing values, data span, record count
* Trend charts:

  * Entropy over time
  * Confidence over time
* Correlation heatmaps
* Z-score anomaly detection
* Per-model statistics (ranker / planner / grounder)
* Configuration comparison leaderboard
* ANOVA significance testing

---

### **5. 🚀 Run Experiment**

Configure and execute the OWG pipeline:

* Set environment parameters:

  * Random seed
  * Number of objects
  * User query
* Per-stage configuration:

  * Grounding
  * Planning
  * Grasping
* For each stage:

  * Enable/disable
  * Select prompt file
  * Edit prompt templates
  * Select model & parameters (temperature, logprobs, max_tokens, n)
* Save final YAML config to:

  ```
  config/pyb/user_defined/config_<timestamp>.yaml
  ```
* Execute pipeline via:

  ```
  notebooks/owg_evaluation_pipeline.py
  ```
* View stdout/stderr output inside Streamlit
* Manage LiteLLM:

  * Check server status
  * Start LiteLLM

---

## **📁 Directory Structure**

```
OWG/
│
├── app.py                          # Streamlit dashboard
│
├── logs/
│   ├── litellm_logs.jsonl          # LiteLLM request logs
│   ├── experiment_metrics.jsonl    # Grasp attempt logs
│   ├── uncertainty_logs_test.jsonl # Uncertainty metadata logs
│
├── prompts/
│   ├── uncertainty_aware/          # System prompts
│   └── user_defined/               # User-created prompts
│
├── config/
│   └── pyb/
│       └── user_defined/           # Saved YAML configs
│
├── notebooks/
│   └── owg_evaluation_pipeline.py  # Main experiment pipeline
│
└── output/                         # Pipeline outputs
```

---

## **▶️ Running the Dashboard**

### **1. Install dependencies**

```bash
pip install -r requirements.txt
```

### **2. Launch Streamlit**

```bash
streamlit run app.py
```

### **3. Start LiteLLM (optional, if not started via UI)**

```bash
litellm --config config/litellm/config.yaml
```

---

## **🔁 Data Flow Diagram (Mermaid)**

```mermaid
graph LR
    subgraph UI["🖥️ Streamlit Dashboard"]
        direction TB
        T1["📊 Tab 1: Logs<br/>(Filters & Charts)"]
        T2["✏️ Tab 2: Prompts<br/>(Browse & Edit)"]
        T3["📈 Tab 3: Metrics<br/>(Success Rates)"]
        T4["🧠 Tab 4: Uncertainty<br/>(Analysis & Stats)"]
        T5["⚙️ Tab 5: Run<br/>(Config & Execute)"]
    end
    
    subgraph DATA["💾 Data Layer"]
        direction TB
        LOG1[("litellm_logs")]
        LOG2[("metrics")]
        LOG3[("uncertainty")]
    end
    
    subgraph FILES["📁 File System"]
        direction TB
        PROMPTS["Prompts<br/>(Base + User)"]
        CONFIG["YAML Configs"]
    end
    
    subgraph EXT["🌐 External Services"]
        direction TB
        LITELLM{{"LiteLLM<br/>:4000"}}
        API[["AI APIs"]]
    end
    
    subgraph EXEC["🔧 Execution"]
        direction TB
        SCRIPT["Pipeline Script"]
        PYB["PyBullet Sim"]
    end
    
    
    %% Read Data Flows
    LOG1 -.->|Read| T1
    LOG2 -.->|Read| T3
    LOG3 -.->|Read| T4
    PROMPTS -.->|Read| T2
    
    %% Write Data Flows
    T2 -->|Save/Delete| PROMPTS
    T5 -->|Generate| CONFIG
    
    %% Execution Flow
    T5 ==>|Execute| SCRIPT
    SCRIPT -->|Control| LITELLM
    LITELLM <-->|API Calls| API
    SCRIPT -->|Simulate| PYB
    
    %% Results Flow
    SCRIPT ==>|Write| LOG1
    SCRIPT ==>|Write| LOG2
    SCRIPT ==>|Write| LOG3
    
    %% Styling
    classDef ui fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    classDef data fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef files fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef external fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef exec fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class UI,T1,T2,T3,T4,T5 ui
    class DATA,LOG1,LOG2,LOG3 data
    class FILES,PROMPTS,CONFIG files
    class EXT,LITELLM,API external
    class EXEC,SCRIPT,PYB exec
```

---

## **📌 Summary**

The OWG Dashboard centralizes:

* Experiment configuration
* Prompt engineering
* LLM log analysis
* Uncertainty metrics
* Grasp performance analytics
* Full pipeline execution

This enables fast experimentation, reproducibility, and optimization of grasping performance in open-world robotic tasks.
