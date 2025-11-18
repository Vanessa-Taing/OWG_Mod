import json
import random
from datetime import datetime, timedelta

def generate_test_logs(num_records=50, output_file="logs/uncertainty_logs_test.jsonl"):
    """
    Generate comprehensive test logs for uncertainty analysis dashboard.
    Creates diverse scenarios including:
    - Multiple model combinations
    - Varying entropy and confidence values
    - Different temporal patterns
    - Some anomalies for testing
    """
    
    # Model options
    models = {
        "ranker": ["gpt-4o", "gpt-4-turbo", "claude-3-opus"],
        "planner": ["gpt-4o", "gpt-4-turbo", "claude-3-sonnet"],
        "grounder": ["gpt-4o", "claude-3-opus", "gpt-4-turbo"]
    }
    
    # Prompt variant options
    prompt_variants = {
        "ranker": [["_uncertainty_description"], ["_baseline"], ["_detailed"]],
        "planner": [["_confidence"], ["_baseline"], ["_step_by_step"]],
        "grounder": [["_cautious"], ["_baseline"], ["_precise"]]
    }
    
    base_time = datetime.now() - timedelta(days=30)
    
    logs = []
    
    for i in range(num_records):
        # Generate timestamp (spread over 30 days)
        timestamp = base_time + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Select model configuration
        ranker_model = random.choice(models["ranker"])
        planner_model = random.choice(models["planner"])
        grounder_model = random.choice(models["grounder"])
        
        # Select prompt variants
        ranker_variants = random.choice(prompt_variants["ranker"])
        planner_variants = random.choice(prompt_variants["planner"])
        grounder_variants = random.choice(prompt_variants["grounder"])
        
        # Generate number of steps (1-3)
        num_steps = random.randint(1, 3)
        
        # Generate metadata for each component
        def generate_component_metadata(component_type, num_steps, is_anomaly=False):
            """Generate metadata for ranker, planner, or grounder"""
            metadata = []
            
            for step in range(num_steps):
                # Base entropy and confidence
                if component_type == "ranker":
                    base_entropy = random.uniform(0.5, 1.5)
                    confidence = -1.0  # Ranker doesn't use confidence
                elif component_type == "planner":
                    base_entropy = random.uniform(0.3, 1.2)
                    confidence = random.uniform(0.7, 0.99)
                else:  # grounder
                    base_entropy = random.uniform(0.4, 1.3)
                    confidence = -1.0
                
                # Add anomalies (10% chance)
                if is_anomaly or random.random() < 0.1:
                    base_entropy *= random.uniform(2.0, 3.5)  # Spike in entropy
                    if confidence > 0:
                        confidence *= random.uniform(0.3, 0.6)  # Drop in confidence
                
                # Generate posterior distribution (2 responses)
                response_1 = f"Response variant A for {component_type} step {step}"
                response_2 = f"Response variant B for {component_type} step {step}"
                
                # Probability split
                prob_1 = random.uniform(0.4, 0.6)
                prob_2 = 1.0 - prob_1
                
                posterior = {
                    response_1: prob_1,
                    response_2: prob_2
                }
                
                uncertainty_descriptions = [
                    "No ambiguity.",
                    "Slight uncertainty in object identification.",
                    "Multiple valid interpretations possible.",
                    "Ambiguous scene layout.",
                    "N/A"
                ]
                
                entry = {
                    "confidence": confidence,
                    "uncertainty_description": random.choice(uncertainty_descriptions),
                    "posterior": posterior,
                    "entropy": base_entropy,
                    "step": step
                }
                
                metadata.append(entry)
            
            return metadata
        
        # Decide if this record should have anomalies
        is_anomaly_record = random.random() < 0.15  # 15% anomaly rate
        
        # Generate metadata for all components
        ranker_metadata = generate_component_metadata("ranker", num_steps, is_anomaly_record)
        planner_metadata = generate_component_metadata("planner", num_steps, is_anomaly_record)
        grounder_metadata = generate_component_metadata("grounder", num_steps, is_anomaly_record)
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "metadata": {
                "ranker": ranker_metadata,
                "planner": planner_metadata,
                "grounder": grounder_metadata
            },
            "model": {
                "ranker": {
                    "model_name": ranker_model,
                    "temperature": 0.0 if ranker_model == "gpt-4o" else 0.1,
                    "max_tokens": 256,
                    "n": 2
                },
                "planner": {
                    "model_name": planner_model,
                    "temperature": 0.0 if planner_model == "gpt-4o" else 0.1,
                    "max_tokens": 256,
                    "n": 2
                },
                "grounder": {
                    "model_name": grounder_model,
                    "temperature": 0.1,
                    "max_tokens": 256,
                    "n": 2
                }
            },
            "prompt_variants": {
                "ranker": ranker_variants,
                "planner": planner_variants,
                "grounder": grounder_variants
            }
        }
        
        logs.append(log_entry)
    
    # Sort by timestamp
    logs.sort(key=lambda x: x["timestamp"])
    
    # Write to file
    with open(output_file, "w") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")
    
    print(f"✅ Generated {num_records} test log entries")
    print(f"📁 Saved to: {output_file}")
    print(f"\n📊 Statistics:")
    print(f"   - Time range: {logs[0]['timestamp']} to {logs[-1]['timestamp']}")
    print(f"   - Unique ranker models: {len(set(l['model']['ranker']['model_name'] for l in logs))}")
    print(f"   - Unique planner models: {len(set(l['model']['planner']['model_name'] for l in logs))}")
    print(f"   - Unique grounder models: {len(set(l['model']['grounder']['model_name'] for l in logs))}")
    
    # Print sample entry
    print(f"\n📝 Sample entry:")
    print(json.dumps(logs[0], indent=2)[:500] + "...")
    
    return logs

# Generate the logs
if __name__ == "__main__":
    # Generate 50 records for comprehensive testing
    logs = generate_test_logs(num_records=50)
    
    # Optional: Generate a smaller set for quick testing
    # logs_small = generate_test_logs(num_records=10, output_file="uncertainty_logs_small.jsonl")