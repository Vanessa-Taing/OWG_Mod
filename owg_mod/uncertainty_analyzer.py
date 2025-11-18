import re
from collections import Counter
from scipy.stats import entropy
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("all-MiniLM-L6-v2")

class UncertaintyAnalyzer:
    @staticmethod
    def calculate_posterior(responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate posterior probabilities across multiple model completions.
        Uses hybrid weighting:
        - Logprob-based when available
        - Semantic similarity-based when unavailable
        - Uniform fallback otherwise
        """
        # Normalize input
        if isinstance(responses[0], str):
            texts = responses
            logprobs = None
        elif isinstance(responses[0], dict):
            texts = [r.get("text", "") for r in responses]
            logprobs = [r.get("avg_logprob") for r in responses if "avg_logprob" in r]
        else:
            raise TypeError("Unsupported response format. Must be list of str or list of dicts.")

        # 1️⃣ Case A: Logprob-based weighting (preferred)
        logprobs = [r.get("avg_logprob") for r in responses if "avg_logprob" in r]
        if logprobs and all(lp is not None for lp in logprobs):
            logprobs = np.array(logprobs)
            exp_probs = np.exp(logprobs - np.max(logprobs))  # numerical stability
            probs = exp_probs / exp_probs.sum()
            posterior = dict(zip(texts, probs.tolist()))
            return posterior

        # 2️⃣ Case B: Semantic similarity weighting
        if len(texts) > 1:
            embeddings = embedder.encode(texts, convert_to_tensor=True)
            mean_embedding = embeddings.mean(dim=0, keepdim=True)
            similarities = util.cos_sim(embeddings, mean_embedding).squeeze().cpu().numpy()
            exp_probs = np.exp(similarities - np.max(similarities))
            probs = exp_probs / exp_probs.sum()
            posterior = dict(zip(texts, probs.tolist()))
            return posterior

        # 3️⃣ Case C: Fallback — uniform weighting
        probs = np.ones(len(texts)) / len(texts)
        return dict(zip(texts, probs.tolist()))

    @staticmethod
    def compute_entropy(posterior: Dict[str, float]) -> float:
        """Compute Shannon entropy (bits) from posterior probabilities."""
        probs = np.array(list(posterior.values()))
        probs = probs[probs > 0]
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


# completions = [
# """To remove nails, you would typically use a hammer with a claw. In the images, the hammer is located at the bottom left.

# - The hammer is marked with ID 16.

# My final answer is: [16]"""
# ,
# """To remove nails, you would typically use a hammer with a claw. In the images, the hammer is located at the bottom left.

# - The hammer is labeled with ID 16.

# My final answer is: [16]"""
# ,"""To remove nails, you would typically use a hammer with a claw. In the images, the hammer is located at the bottom left.

# - The hammer is labeled with ID 16.

# My final answer is: [16]""",

# """To remove nails, you would typically use a hammer with a claw. In the images, the hammer is located at the bottom left.

# - The hammer is labeled with ID 16.

# My final answer is: [16]""", """To remove nails, you would typically use a hammer with a claw. In the images, the hammer is located at the bottom left.

# - The hammer is labeled with ID 16.

# My final answer is: [16]"""
# ]

# posterior = UncertaintyAnalyzer.calculate_posterior(completions)
# H = UncertaintyAnalyzer.calculate_entropy(posterior)

# print("Posterior:", posterior)
# print(f"Entropy over completions: {H:.2f} bits")