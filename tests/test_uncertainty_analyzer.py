import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path so that `owg_mod` can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from owg_mod import uncertainty_analyzer as ua  # noqa: E402


class DummyEmbedder:
    """Simple stand-in for SentenceTransformer used in tests.

    It returns deterministic embeddings based on the index only, so that
    clustering and similarity computations are stable and cheap.
    """

    def encode(self, texts, convert_to_tensor=True):
        n = len(texts)
        # Use an identity-like matrix so each text is orthogonal to others
        return np.eye(n, dtype=np.float32)


@pytest.fixture(autouse=True)
def patch_embedder(monkeypatch):
    """Automatically patch the global embedder in uncertainty_analyzer for all tests."""
    monkeypatch.setattr(ua, "embedder", DummyEmbedder())
    yield


def test_cluster_equivalent_responses_single_text():
    labels = ua.UncertaintyAnalyzer.cluster_equivalent_responses(["only one"], np.array([[1.0]]))
    assert np.array_equal(labels, np.array([0]))


def test_cluster_equivalent_responses_multiple_texts():
    texts = ["a", "b", "c"]
    # Use deterministic embeddings (3x3 identity)
    embeddings = np.eye(3, dtype=np.float32)

    labels = ua.UncertaintyAnalyzer.cluster_equivalent_responses(texts, embeddings, threshold=0.5)

    # With orthogonal embeddings and moderately high threshold, each point should form its own cluster
    assert len(labels) == 3
    assert set(labels) == {0, 1, 2}


def test_calculate_posterior_with_logprobs():
    # Two semantically distinct responses with explicit logprobs
    responses = [
        {"text": "answer A", "avg_logprob": -0.1},
        {"text": "answer B", "avg_logprob": -2.0},
    ]

    posterior = ua.UncertaintyAnalyzer.calculate_posterior(responses, cluster_threshold=0.0)

    # Probabilities should sum to ~1
    total_prob = sum(posterior.values())
    assert pytest.approx(total_prob, rel=1e-6) == 1.0

    # answer A has much higher logprob -> higher posterior mass
    assert posterior["answer A"] > posterior["answer B"]


def test_calculate_posterior_without_logprobs():
    # Pure text responses; DummyEmbedder makes them orthogonal, so they each form own cluster
    texts = ["alpha", "beta"]

    posterior = ua.UncertaintyAnalyzer.calculate_posterior(texts, cluster_threshold=0.0)

    total_prob = sum(posterior.values())
    assert pytest.approx(total_prob, rel=1e-6) == 1.0
    assert set(posterior.keys()) == {"alpha", "beta"}


def test_compute_entropy_basic():
    posterior = {"a": 0.5, "b": 0.5}
    ent = ua.UncertaintyAnalyzer.compute_entropy(posterior)
    # Entropy of a fair coin is 1 bit
    assert pytest.approx(ent, rel=1e-6) == 1.0

    posterior_peaked = {"a": 0.9, "b": 0.1}
    ent_peaked = ua.UncertaintyAnalyzer.compute_entropy(posterior_peaked)
    assert ent_peaked < ent


def test_extract_final_answer_key_matches_pattern_and_fallback():
    text = "The final answer is: [1, 2, 3]"
    key = ua.UncertaintyAnalyzer.extract_final_answer_key(text)
    assert key == "[1, 2, 3]"

    # Fallback to entire text if pattern not found
    raw = "No explicit final answer marker here."
    assert ua.UncertaintyAnalyzer.extract_final_answer_key(raw) == raw


@pytest.mark.parametrize(
    "resp, expected",
    [
        ("confidence: 0.85", 0.85),
        ("Confidence score: 0.7.", 0.7),
        ("no confidence here", None),
    ],
)
def test_parse_confidence_from_response(resp, expected):
    assert ua.UncertaintyAnalyzer.parse_confidence_from_response(resp) == expected


@pytest.mark.parametrize(
    "resp, expected",
    [
        ('Uncertainty: "object is occluded by another object"', "object is occluded by another object"),
        ("Uncertainty: high occlusion\nNext line", "high occlusion"),
        ("No uncertainty section", None),
    ],
)
def test_extract_uncertainty_descriptors(resp, expected):
    assert ua.UncertaintyAnalyzer.extract_uncertainty_descriptors(resp) == expected


def test_extract_metadata_uses_defaults_and_parsed_values():
    resp = 'confidence: 0.6\nUncertainty: "target is partially covered"'
    meta = ua.UncertaintyAnalyzer.extract_metadata(resp)

    assert meta["confidence"] == 0.6
    assert meta["uncertainty_description"] == "target is partially covered"

    # When nothing is present, fall back to defaults
    empty_meta = ua.UncertaintyAnalyzer.extract_metadata("no relevant info")
    assert empty_meta["confidence"] == -1.0
    assert empty_meta["uncertainty_description"] == "N/A"



