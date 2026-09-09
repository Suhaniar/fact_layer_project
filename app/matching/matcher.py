import os
import numpy as np

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Embedding model
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)


# Similarity threshold
SIMILARITY_THRESHOLD = float(
    os.getenv(
        "SIMILARITY_THRESHOLD",
        "0.62"
    )
)


class FactMatcher:

    def __init__(self):

        self.model = SentenceTransformer(
            MODEL_NAME
        )

    def fact_to_text(self, fact):

        subject = fact.get(
            "subject",
            ""
        )

        predicate = fact.get(
            "predicate",
            ""
        )

        value = fact.get(
            "value_text",
            ""
        )

        unit = fact.get(
            "unit",
            ""
        )

        scope = fact.get(
            "scope",
            ""
        )

        location = fact.get(
            "location",
            ""
        )

        period = fact.get(
            "period",
            ""
        )

        return (
            f"subject: {subject}; "
            f"predicate: {predicate}; "
            f"value: {value}; "
            f"unit: {unit}; "
            f"period: {period}; "
            f"scope: {scope}; "
            f"location: {location}"
        )

    def create_embeddings(self, facts):

        texts = [
            self.fact_to_text(fact)
            for fact in facts
        ]

        if not texts:

            return np.array([])

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings

    def find_candidates(self, facts):

        if len(facts) < 2:

            return []

        embeddings = self.create_embeddings(
            facts
        )

        candidates = []

        for i in range(len(facts)):

            for j in range(i + 1, len(facts)):

                # Only compare facts from different documents
                document_a = facts[i].get(
                    "document_name"
                )

                document_b = facts[j].get(
                    "document_name"
                )

                if document_a == document_b:
                    continue

                # Since embeddings are normalized,
                # dot product = cosine similarity
                similarity = float(
                    np.dot(
                        embeddings[i],
                        embeddings[j]
                    )
                )

                if similarity >= SIMILARITY_THRESHOLD:

                    candidates.append({

                        "fact_a_index": i,

                        "fact_b_index": j,

                        "similarity": similarity
                    })

        # Highest similarity first
        candidates.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        return candidates


# ---------------------------------------------------------
# Compatibility function used by pipeline.py
# ---------------------------------------------------------

def find_candidate_pairs(
    facts,
    embeddings=None,
    threshold=SIMILARITY_THRESHOLD
):
    """
    Find semantically similar fact pairs from different
    documents.

    This function is kept separately because pipeline.py
    imports it directly:

        from app.matching.matcher import find_candidate_pairs

    Returns:
        [
            {
                "fact_a": fact,
                "fact_b": fact,
                "similarity": float
            }
        ]
    """

    # Need at least two facts
    if len(facts) < 2:

        return []

    # If embeddings were not provided,
    # create them using FactMatcher
    if embeddings is None:

        matcher = FactMatcher()

        embeddings = matcher.create_embeddings(
            facts
        )

    # Convert embeddings to numpy array
    embeddings = np.asarray(
        embeddings
    )

    # Make sure the number of embeddings
    # matches the number of facts
    if len(embeddings) != len(facts):

        raise ValueError(
            f"Number of embeddings ({len(embeddings)}) "
            f"does not match number of facts ({len(facts)})."
        )

    # Normalize embeddings
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    # Prevent division by zero
    norms[norms == 0] = 1e-8

    normalized_embeddings = (
        embeddings / norms
    )

    # Calculate complete cosine similarity matrix
    similarity_matrix = (
        normalized_embeddings
        @ normalized_embeddings.T
    )

    candidates = []

    # Compare every unique pair
    for i in range(len(facts)):

        for j in range(i + 1, len(facts)):

            # Only compare facts from different documents
            document_a = facts[i].get(
                "document_name"
            )

            document_b = facts[j].get(
                "document_name"
            )

            if document_a == document_b:

                continue

            similarity = float(
                similarity_matrix[i, j]
            )

            # Keep only sufficiently similar facts
            if similarity >= threshold:

                candidates.append({

                    "fact_a": facts[i],

                    "fact_b": facts[j],

                    "similarity": similarity
                })

    # Sort from highest similarity to lowest
    candidates.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return candidates