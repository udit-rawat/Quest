import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import os
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Solution:
    title: str
    solution: str
    difficulty: str
    topics: str
    companies: str


class LeetCodeRetriever:
    def __init__(
        self,
        index_path: str = os.path.join(current_dir, "leetcode_hnsw.index"),
        metadata_path: str = os.path.join(
            current_dir, "leetcode_metadata.pkl"),
        model_name: str = "all-MiniLM-L6-v2",
        ef_search: int = 32  # Adjust for speed/accuracy trade-off
    ):
        """Initialize the retriever with an HNSW index and metadata."""
        self.encoder = SentenceTransformer(model_name)

        # Load the HNSW index
        self.index = faiss.read_index(index_path)

        # Ensure the index is an HNSW index
        if not isinstance(self.index, faiss.IndexHNSWFlat):
            raise ValueError(
                "The provided index is not an HNSW index. Please recreate the index using IndexHNSWFlat.")

        # Set HNSW-specific parameters
        self.index.hnsw.efSearch = ef_search  # Tune for faster retrieval

        # Load metadata
        self.solutions = self._load_metadata(metadata_path)
        logger.info("Retriever initialized successfully.")

    def _load_metadata(self, metadata_path: str) -> List[Solution]:
        """Load metadata from a pickle file."""
        try:
            with open(metadata_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise

    def search(
        self,
        query: str,
        k: int = 3,
        return_scores: bool = True
    ) -> List[Tuple[Solution, float]]:
        """Search for similar solutions using the HNSW index."""
        try:
            # Encode query
            query_vector = self.encoder.encode(
                [query], show_progress_bar=False)
            query_vector = query_vector.astype(np.float32)

            # Search index
            distances, indices = self.index.search(query_vector, k)

            # Return results
            if return_scores:
                return [
                    (self.solutions[idx], float(score))
                    for idx, score in zip(indices[0], distances[0])
                ]
            return [self.solutions[idx] for idx in indices[0]]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_topic_similar(
        self,
        topic: str,
        k: int = 5
    ) -> List[Solution]:
        """Find solutions with similar topics."""
        try:
            return [
                sol for sol in self.solutions
                if topic.lower() in sol.topics.lower()
            ][:k]
        except Exception as e:
            logger.error(f"Topic search failed: {e}")
            return []


if __name__ == "__main__":
    retriever = LeetCodeRetriever()
    queries = [
        "matrix distance calculation",
        "dynamic programming coin change",
        "tree traversal problem"
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.search(query, k=2)
        for solution, score in results:
            print(f"\nTitle: {solution.title}")
            print(f"Score: {score:.3f}")
            print(f"Topics: {solution.topics}")
            print(f"Difficulty: {solution.difficulty}")
