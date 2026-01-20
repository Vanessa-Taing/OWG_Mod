"""
Comprehensive demonstration of UncertaintyAnalyzer showing BOTH cases:
- Case A: When log-probabilities are available
- Case B: When log-probabilities are NOT available (fallback to similarity)
"""

import re
import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import util, SentenceTransformer

# Use your existing code's embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')


class UncertaintyAnalyzer:
    """Exact copy of your original class - unchanged logic"""
    
    @staticmethod
    def cluster_equivalent_responses(texts: List[str], embeddings, threshold: float = 0.85):
        if len(texts) == 1:
            return np.array([0])
        
        sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        dist_matrix = 1 - sim_matrix
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - threshold,
            metric='precomputed',
            linkage='complete'
        )
        labels = clustering.fit_predict(dist_matrix)
        return labels
    
    @staticmethod
    def calculate_posterior(responses: List[Dict[str, Any]],
                          cluster_threshold: float = 0.85) -> Dict[str, float]:
        # Normalize input
        if isinstance(responses[0], str):
            texts = responses
            logprobs = [None] * len(texts)
        elif isinstance(responses[0], dict):
            texts = [r.get("text", "") for r in responses]
            logprobs = [r.get("avg_logprob") for r in responses]
        else:
            raise TypeError("Unsupported response format")
        
        # Step 1: Cluster semantically equivalent responses
        embeddings = embedder.encode(texts, convert_to_tensor=True)
        cluster_labels = UncertaintyAnalyzer.cluster_equivalent_responses(
            texts, embeddings, threshold=cluster_threshold
        )
        
        # Step 2: Aggregate within clusters
        n_clusters = len(set(cluster_labels))
        cluster_weights = np.zeros(n_clusters)
        cluster_to_canonical = {}
        
        # Case A: Use logprobs if available for all responses
        if all(lp is not None for lp in logprobs):
            lp_array = np.array(logprobs)
            weights = np.exp(lp_array - lp_array.max())
            
            for cluster_id in range(n_clusters):
                mask = cluster_labels == cluster_id
                cluster_weights[cluster_id] = weights[mask].sum()
                cluster_indices = np.where(mask)[0]
                best_idx = cluster_indices[np.argmax(lp_array[mask])]
                cluster_to_canonical[cluster_id] = texts[best_idx]
        
        # Case B: Use semantic similarity to centroid
        else:
            for cluster_id in range(n_clusters):
                mask = cluster_labels == cluster_id
                cluster_embeddings = embeddings[mask]
                centroid = cluster_embeddings.mean(dim=0, keepdim=True)
                similarities = util.cos_sim(cluster_embeddings, centroid).squeeze()
                if similarities.dim() == 0:
                    similarities = similarities.unsqueeze(0)
                
                cluster_weights[cluster_id] = similarities.sum().cpu().item()
                cluster_indices = np.where(mask)[0]
                best_idx = cluster_indices[similarities.argmax().cpu().item()]
                cluster_to_canonical[cluster_id] = texts[best_idx]
        
        # Step 3: Normalize to get probabilities
        cluster_probs = cluster_weights / cluster_weights.sum()
        
        # Step 4: Build posterior with canonical texts
        posterior = {}
        for cluster_id in range(n_clusters):
            canonical_text = cluster_to_canonical[cluster_id]
            posterior[canonical_text] = float(cluster_probs[cluster_id])
        
        return posterior
    
    @staticmethod
    def compute_entropy(posterior: Dict[str, float]) -> float:
        probs = np.array(list(posterior.values()))
        probs = probs[probs > 0]
        
        if len(probs) == 0:
            return 0.0
        
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)


def print_computation_details(responses: List[Dict[str, Any]], name: str, case_label: str):
    """Print concise computation breakdown"""
    
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  {case_label}")
    print(f"{'='*70}")
    
    # Extract data
    if isinstance(responses[0], str):
        texts = responses
        logprobs = [None] * len(texts)
    else:
        texts = [r.get("text", "") for r in responses]
        logprobs = [r.get("avg_logprob") for r in responses]
    
    # Determine which case will be used
    has_logprobs = all(lp is not None for lp in logprobs)
    
    # Show inputs
    print(f"\n📝 Inputs: {len(responses)} responses")
    for i, (t, lp) in enumerate(zip(texts, logprobs)):
        lp_str = f"{lp:.3f}" if lp is not None else "None"
        print(f"  [{i}] logprob={lp_str} | {t[:50]}...")
    
    # Clustering
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    cluster_labels = UncertaintyAnalyzer.cluster_equivalent_responses(texts, embeddings)
    
    print(f"\n🔗 Clustering: {len(set(cluster_labels))} clusters")
    print(f"  Similarity matrix (cosine):")
    for row in sim_matrix:
        print(f"    {' '.join([f'{x:.3f}' for x in row])}")
    print(f"  Cluster assignments: {cluster_labels}")
    
    # Weight computation differs by case
    n_clusters = len(set(cluster_labels))
    
    if has_logprobs:
        print(f"\n⚖️  CASE A: Using log-probability weighting")
        print(f"  Weights: exp(logprob - max_logprob)")
        
        lp_array = np.array(logprobs)
        weights = np.exp(lp_array - lp_array.max())
        
        for i, (lp, w) in enumerate(zip(lp_array, weights)):
            print(f"  [{i}] exp({lp:.3f} - {lp_array.max():.3f}) = {w:.6f}")
        
        # Aggregate by cluster
        cluster_weights = np.zeros(n_clusters)
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            cluster_weights[cluster_id] = weights[mask].sum()
        
    else:
        print(f"\n⚖️  CASE B: Using semantic similarity weighting")
        print(f"  Weights: similarity to cluster centroid")
        
        cluster_weights = np.zeros(n_clusters)
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            cluster_embeddings = embeddings[mask]
            centroid = cluster_embeddings.mean(dim=0, keepdim=True)
            similarities = util.cos_sim(cluster_embeddings, centroid).squeeze()
            if similarities.dim() == 0:
                similarities = similarities.unsqueeze(0)
            
            sim_vals = similarities.cpu().numpy()
            cluster_weights[cluster_id] = similarities.sum().cpu().item()
            
            members = np.where(mask)[0]
            print(f"  Cluster {cluster_id}:")
            for idx, sim in zip(members, sim_vals):
                print(f"    [{idx}] similarity to centroid = {sim:.6f}")
            print(f"    Total weight = {cluster_weights[cluster_id]:.6f}")
    
    print(f"\n📊 Cluster Aggregation:")
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        members = np.where(mask)[0].tolist()
        print(f"  Cluster {cluster_id}: responses {members} → weight={cluster_weights[cluster_id]:.6f}")
    
    # Normalize
    cluster_probs = cluster_weights / cluster_weights.sum()
    
    print(f"\n🎯 Posterior Probabilities:")
    posterior = UncertaintyAnalyzer.calculate_posterior(responses)
    for i, (text, prob) in enumerate(posterior.items()):
        print(f"  P(cluster {i}) = {prob:.6f}")
    
    # Entropy
    entropy = UncertaintyAnalyzer.compute_entropy(posterior)
    max_entropy = np.log2(len(posterior))
    
    print(f"\n📈 Entropy Calculation:")
    print(f"  H = -Σ p(x)·log₂(p(x))")
    for i, (text, p) in enumerate(posterior.items()):
        term = -p * np.log2(p) if p > 0 else 0
        print(f"    Term {i}: -{p:.6f} × log₂({p:.6f}) = {term:.6f}")
    
    print(f"  Total: H = {abs(entropy):.6f} bits")
    
    if max_entropy > 0:
        print(f"  Range: [0, {max_entropy:.3f}] | Current: {entropy/max_entropy*100:.1f}% of max")
    else:
        print(f"  Range: [0, 0.000] | Single cluster (deterministic)")
    
    return posterior, entropy


def main():
    """Comprehensive demo showing BOTH cases"""
    
    print("\n" + "="*70)
    print("  UNCERTAINTY ANALYZER - Comprehensive Demonstration")
    print("="*70)
    print("""
  This demo showcases TWO cases:
  
  CASE A: Log-probabilities available
    - Prior: Uniform over responses
    - Likelihood: exp(avg_logprob) 
    - Weight: Based on model confidence
    
  CASE B: Log-probabilities NOT available (fallback)
    - Prior: Uniform over responses
    - Weight: Based on semantic similarity to cluster centroid
    - Useful when logprobs are unavailable (e.g., API limitations)
    """)
    
    # =========================================================================
    # CASE A DEMONSTRATIONS (with logprobs)
    # =========================================================================
    
    print("\n" + "█"*70)
    print("  CASE A: LOG-PROBABILITIES AVAILABLE")
    print("█"*70)
    
    # Demo 1A: Identical responses with logprobs
    responses_identical_with_logprobs = [
        {"text": "The smallest is strawberry (Object 11). Final: [11]", 
         "avg_logprob": -0.5},
        {"text": "Object 11 (strawberry) is smallest. Answer: [11]", 
         "avg_logprob": -0.6},
        {"text": "Strawberry (11) is the smallest. Final: [11]",
         "avg_logprob": -0.55}
    ]
    
    post1a, ent1a = print_computation_details(
        responses_identical_with_logprobs, 
        "DEMO 1A: Identical Responses (with logprobs)",
        "CASE A: Using exp(logprob) weighting"
    )
    
    # Demo 2A: Different responses with logprobs
    responses_different_with_logprobs = [
        {"text": "The smallest is strawberry (Object 11). Final: [11]", 
         "avg_logprob": -0.5},
        {"text": "The smallest is the lid (Object 12). Final: [12]", 
         "avg_logprob": -0.6},
        {"text": "I believe it's marble (Object 13). Answer: [13]",
         "avg_logprob": -0.8}
    ]
    
    post2a, ent2a = print_computation_details(
        responses_different_with_logprobs, 
        "DEMO 2A: Different Responses (with logprobs)",
        "CASE A: Using exp(logprob) weighting"
    )
    
    # =========================================================================
    # CASE B DEMONSTRATIONS (without logprobs)
    # =========================================================================
    
    print("\n" + "█"*70)
    print("  CASE B: LOG-PROBABILITIES NOT AVAILABLE (Fallback Mode)")
    print("█"*70)
    
    # Demo 1B: Identical responses WITHOUT logprobs
    responses_identical_no_logprobs = [
        "The smallest is strawberry (Object 11). Final: [11]",
        "Object 11 (strawberry) is smallest. Answer: [11]",
        "Strawberry (11) is the smallest. Final: [11]"
    ]
    
    post1b, ent1b = print_computation_details(
        responses_identical_no_logprobs, 
        "DEMO 1B: Identical Responses (NO logprobs)",
        "CASE B: Using semantic similarity weighting"
    )
    
    # Demo 2B: Different responses WITHOUT logprobs
    responses_different_no_logprobs = [
        "The smallest is strawberry (Object 11). Final: [11]",
        "The smallest is the lid (Object 12). Final: [12]",
        "I believe it's marble (Object 13). Answer: [13]"
    ]
    
    post2b, ent2b = print_computation_details(
        responses_different_no_logprobs, 
        "DEMO 2B: Different Responses (NO logprobs)",
        "CASE B: Using semantic similarity weighting"
    )
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    
    print(f"\n{'='*70}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nCASE A (with logprobs):")
    print(f"  Demo 1A (identical):  Entropy = {abs(ent1a):.6f} bits → High confidence")
    print(f"  Demo 2A (different):  Entropy = {abs(ent2a):.6f} bits → High uncertainty")
    
    print(f"\nCASE B (without logprobs - similarity fallback):")
    print(f"  Demo 1B (identical):  Entropy = {abs(ent1b):.6f} bits → High confidence")
    print(f"  Demo 2B (different):  Entropy = {abs(ent2b):.6f} bits → High uncertainty")
    
    print(f"\n{'='*70}")
    print("  KEY INSIGHTS")
    print(f"{'='*70}")
    print("""
  1. CASE A (logprobs available):
     - Uses model's internal confidence (exp(logprob))
     - More accurate uncertainty quantification
     - Reflects both semantic AND model confidence
  
  2. CASE B (logprobs unavailable):
     - Fallback to pure semantic similarity
     - Still provides meaningful uncertainty estimates
     - Useful for API-only access or external models
  
  3. Both cases produce similar entropy patterns:
     - Identical responses → Low entropy (certain)
     - Different responses → High entropy (uncertain)
     
  4. Case A is preferred when available (better calibration)
     Case B ensures robustness when logprobs are missing
  """)
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()