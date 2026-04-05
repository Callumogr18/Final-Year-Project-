import os
import math
import logging
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename="hybrid_eval.log"
)

TASK_WEIGHTS = {
    'QA': {
        'alpha': 0.7,
        'beta': 0.3,
        'gamma': 0.5,
        'quality': {
            'fluency': 0.10,
            'coherence': 0.10,
            'consistency': 0.20,
            'reasoning': 0.25,
            'factual_accuracy': 0.35,
        },
    },
    'SUMMARISATION': {
        'alpha': 0.3,
        'beta': 0.7,
        'gamma': 0.5,
        'quality': {
            'fluency': 0.20,
            'coherence': 0.25,
            'consistency': 0.20,
            'reasoning': 0.10,
            'factual_accuracy': 0.25,
        },
    },
}

embedding_client = None


def getembedding_client():
    global embedding_client
    if embedding_client is None:
        embedding_client = OpenAI(
            base_url=os.getenv("EMBEDDING_ENDPOINT", os.getenv("GPT_ENDPOINT")),
            api_key=os.getenv("API_KEY"),
        )
    return embedding_client


def get_embeddings(texts):
    """Get embeddings for a list of texts in a single API call."""
    client = getembedding_client()
    model = os.getenv("EMBEDDING_MODEL")

    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two vectors, clamped to [0, 1]."""
    a = np.array(vec_a)
    b = np.array(vec_b)

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    sim = dot / (norm_a * norm_b)
    return float(max(0.0, min(1.0, sim)))


def compute_quality_score(judge_scores, quality_weights):
    """ Weighted sum of judge quality metrics (excluding hallucination) """
    return sum(
        quality_weights[metric] * judge_scores[metric]
        for metric in quality_weights
    )


def hallucination_gate(hallucination_score, gamma):
    """
    Penalty based on hallucination score.

    Uses H^gamma (gamma=0.5 by default, i.e. square root) so that:
      H=1.0 -> 1.0   (no penalty)
      H=0.5 -> 0.71  (meaningful penalty)
      H=0.0 -> 0.0   (total suppression)
    """
    return math.pow(hallucination_score, gamma)


def compute_hybrid_score(response_text, reference_text, judge_scores, task_type):
    """
    Compute the HybridEval score.

    Formula: HybridEval = Sim^alpha * Q^beta * H^gamma

    Args:
        response_text: the LLM's response string
        reference_text: the ground truth reference string
        judge_scores: dict with keys 'hallucination', 'fluency', 'coherence',
                      'consistency', 'reasoning', 'factual_accuracy'
        task_type: 'QA' or 'SUMMARISATION'

    Returns:
        dict with keys:
            'hybrid_score': final composite score in [0, 1]
            'similarity': cosine similarity between embeddings
            'quality': weighted judge quality score
            'hallucination_gate': the gate value applied
    """
    weights = TASK_WEIGHTS[task_type.upper()]

    embeddings = get_embeddings([response_text, reference_text])
    sim = cosine_similarity(embeddings[0], embeddings[1])
    logger.info(f"Embedding similarity: {sim:.4f}")

    quality = compute_quality_score(judge_scores, weights['quality'])
    logger.info(f"Quality score: {quality:.4f}")

    gate = hallucination_gate(judge_scores['hallucination'], weights['gamma'])
    logger.info(f"Hallucination gate: {gate:.4f}")

    hybrid_score = (sim ** weights['alpha']) * (quality ** weights['beta']) * gate

    logger.info(f"HybridEval = {sim:.4f}^{weights['alpha']} "
                f"* {quality:.4f}^{weights['beta']} "
                f"* {gate:.4f} = {hybrid_score:.4f}")

    return {
        'hybrid_score': hybrid_score,
        'similarity': sim,
        'quality': quality,
        'hallucination_gate': gate,
    }
