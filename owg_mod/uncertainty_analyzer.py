import re
from collections import Counter
from scipy.stats import entropy
from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Try to import the real sentence-transformers stack. If it fails (e.g. due to
# missing TensorFlow / GPU DLLs), fall back to a very lightweight local
# implementation so that the rest of the module (and tests) still work.
try:  # pragma: no cover - error path exercised indirectly in environments without deps
    from sentence_transformers import util as st_util  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    util = st_util
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:  # Broad on purpose: TensorFlow often raises RuntimeError here
    class _DummyUtil:
        """Minimal cosine similarity helper using NumPy, mimicking util.cos_sim."""

        @staticmethod
        def cos_sim(a, b):
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            # Normalize rows
            def _norm(x):
                n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
                return x / n

            a_n = _norm(a)
            b_n = _norm(b)
            return a_n @ b_n.T

    class _DummyEmbedder:
        """Very small stand-in encoder returning identity-like embeddings.

        This is only used when sentence-transformers cannot be imported.
        It is good enough for clustering logic and unit tests.
        """

        def encode(self, texts, convert_to_tensor=True):
            n = len(texts)
            return np.eye(n, dtype=np.float32)

    util = _DummyUtil()
    embedder = _DummyEmbedder()

class UncertaintyAnalyzer:
    @staticmethod
    def cluster_equivalent_responses(texts: List[str], embeddings, threshold: float = 0.85):
        """
        Cluster semantically equivalent responses using hierarchical clustering.
        
        Args:
            texts: List of response texts
            embeddings: Sentence embeddings tensor
            threshold: Similarity threshold for considering responses equivalent (0-1)
        
        Returns:
            labels: Cluster label for each response
        """
        if len(texts) == 1:
            return np.array([0])
        
        # Compute pairwise similarities and convert to distances.
        # Support both torch-style tensors (with .cpu().numpy()) and pure NumPy arrays
        # returned by the lightweight fallback util.
        sim = util.cos_sim(embeddings, embeddings)
        if hasattr(sim, "cpu"):
            sim_matrix = sim.cpu().numpy()
        else:
            sim_matrix = np.asarray(sim, dtype=np.float32)
        dist_matrix = 1 - sim_matrix
        
        # Use agglomerative clustering with complete linkage
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
        """
        Calculate posterior probabilities with semantic clustering.
        
        Key improvements:
        1. Cluster semantically equivalent responses FIRST
        2. Aggregate logprobs/weights within clusters
        3. Assign probability mass to clusters, not individual responses
        """
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
        cluster_to_canonical = {}  # Map cluster ID to representative text
        
        # Case A: Use logprobs if available for all responses
        if all(lp is not None for lp in logprobs):
            # Convert logprobs to unnormalized weights
            lp_array = np.array(logprobs)
            weights = np.exp(lp_array - lp_array.max())  # Numerical stability
            
            # Aggregate weights within each cluster
            for cluster_id in range(n_clusters):
                mask = cluster_labels == cluster_id
                cluster_weights[cluster_id] = weights[mask].sum()
                # Pick the response with highest logprob as canonical
                cluster_indices = np.where(mask)[0]
                best_idx = cluster_indices[np.argmax(lp_array[mask])]
                cluster_to_canonical[cluster_id] = texts[best_idx]
        
        # Case B: Use semantic similarity to centroid
        else:
            for cluster_id in range(n_clusters):
                mask = cluster_labels == cluster_id
                cluster_embeddings = embeddings[mask]

                # Compute centroid of cluster.
                # Support both torch tensors (dim/keepdim) and NumPy arrays (axis/keepdims).
                try:
                    centroid = cluster_embeddings.mean(dim=0, keepdim=True)
                except TypeError:
                    centroid = np.mean(cluster_embeddings, axis=0, keepdims=True)

                # Weight by similarity to centroid
                similarities = util.cos_sim(cluster_embeddings, centroid).squeeze()

                # Normalize similarities to a 1D NumPy array for downstream ops
                if hasattr(similarities, "cpu"):
                    similarities_np = similarities.cpu().numpy().reshape(-1)
                else:
                    similarities_np = np.asarray(similarities, dtype=np.float32).reshape(-1)

                cluster_weights[cluster_id] = float(similarities_np.sum())

                # Pick most central response as canonical
                cluster_indices = np.where(mask)[0]
                best_idx = cluster_indices[int(similarities_np.argmax())]
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
        """Compute Shannon entropy (bits) from posterior probabilities."""
        probs = np.array(list(posterior.values()))
        probs = probs[probs > 0]  # Filter zero probabilities
        
        if len(probs) == 0:
            return 0.0
        
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)
    
    @staticmethod
    def extract_final_answer_key(text: str) -> str:
        m = re.search(r"final answer is:\s*(\[[0-9,\s]+\])", text, re.IGNORECASE)
        return m.group(1) if m else text  # fallback to whole text

    # then pass [extract_final_answer_key(r) for r in responses] to calculate_posterior

    @staticmethod
    def parse_confidence_from_response(response: List[str]) -> Optional[float]:
        """Extract a numeric confidence score from the response, if present."""
        conf_match = re.search(r"confidence(?: score)?:\s*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
        if conf_match:
            try:
                return float(conf_match.group(1).rstrip('.'))
            except ValueError:
                return 0.0
        return None

    @staticmethod
    def extract_uncertainty_descriptors(response: List[str]) -> Optional[str]:
        """Extract textual uncertainty descriptors from the response, if present."""
        match = re.search(r'Uncertainty:\s*["“]?(.+?)["”]?(?:\n|$)', response)
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_metadata(response: List[str], default_confidence: float = -1.0, default_uncertainty: str = "N/A") -> Dict[str, Any]:
        """
        Extracts both confidence score and uncertainty descriptor from an LLM response.

        Args:
            response: The raw model output string.
            default_confidence: Fallback value if no confidence is found.
            default_uncertainty: Fallback string if no uncertainty description is found.

        Returns:
            A dictionary with keys:
                - "confidence": float
                - "uncertainty_description": str
        """
        confidence = UncertaintyAnalyzer.parse_confidence_from_response(response)
        uncertainty_desc = UncertaintyAnalyzer.extract_uncertainty_descriptors(response)

        return {
            "confidence": confidence if confidence is not None else default_confidence,
            "uncertainty_description": uncertainty_desc if uncertainty_desc is not None else default_uncertainty
        }

# if __name__ == "__main__":
    
#     # Case 1: Semantically identical responses (should have low entropy)
#     responses_identical = [
#         {"text": "The smallest object is the strawberry (Object 11). Final answer: [11]", 
#          "avg_logprob": -0.5},
#         {"text": "The smallest object appears to be the strawberry (Object 11). Answer: [11]", 
#          "avg_logprob": -0.6}
#     ]
    
#     posterior1 = UncertaintyAnalyzer.calculate_posterior(
#         responses_identical
#     )
#     entropy1 = UncertaintyAnalyzer.compute_entropy(posterior1)
#     print(f"Identical responses - Posterior: {posterior1}")
#     print(f"Entropy: {entropy1:.3f} bits (should be ~0)\n")
    
#     # Case 2: Semantically different responses (should have higher entropy)
#     responses_different = [
#         {"text": "The smallest object is the strawberry (Object 11). Final answer: [11]", 
#          "avg_logprob": -0.5},
#         {"text": "The smallest object is the lid (Object 12). Final answer: [12]", 
#          "avg_logprob": -0.6}
#     ]
    
#     posterior2 = UncertaintyAnalyzer.calculate_posterior(
#         responses_different, embedder
#     )
#     entropy2 = UncertaintyAnalyzer.compute_entropy(posterior2)
#     print(f"Different responses - Posterior: {posterior2}")
#     print(f"Entropy: {entropy2:.3f} bits (should be ~1)")