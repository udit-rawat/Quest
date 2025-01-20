import pickle
import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import os
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
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.encoder = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)

        with open(metadata_path, 'rb') as f:
            self.solutions = pickle.load(f)

    def search(
        self,
        query: str,
        k: int = 3,
        return_scores: bool = True
    ) -> List[Tuple[Solution, float]]:
        """Search for similar solutions"""
        # Encode query
        query_vector = self.encoder.encode([query])

        # Search index
        distances, indices = self.index.search(
            query_vector.astype(np.float32), k
        )

        # Return results
        if return_scores:
            return [
                (self.solutions[idx], float(score))
                for idx, score in zip(indices[0], distances[0])
            ]
        return [self.solutions[idx] for idx in indices[0]]

    def get_topic_similar(
        self,
        topic: str,
        k: int = 5
    ) -> List[Solution]:
        """Find solutions with similar topics"""
        return [
            sol for sol in self.solutions
            if topic.lower() in sol.topics.lower()
        ][:k]


if __name__ == '__main__':
    retriever = LeetCodeRetriever()
    query = "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target."
    results = retriever.search(query)
    for result in results:
        print(result[0].title)
        print(result[0].solution)
        print(f"Score: {result[1]}")
        print()
