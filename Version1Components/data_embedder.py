import json
import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
import glob
import pickle
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Solution:
    title: str
    solution: str
    difficulty: str
    topics: str
    companies: str


class LeetCodeEmbedder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "leetcode_hnsw.index",
        metadata_path: str = "leetcode_metadata.pkl"
    ):
        self.encoder = SentenceTransformer(model_name)
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.solutions: List[Solution] = []
        self.dimension = 384  # MiniLM-L6-v2 dimension

    def process_solution(self, json_data: dict) -> str:
        """Extract and clean solution text from JSON"""
        solution = json_data.get('solution', '')
        # Remove markdown code blocks
        solution = solution.replace('```python', '').replace('```', '')
        # Extract approach and implementation details
        if 'Approach:' in solution:
            solution = solution.split('Approach:')[1]
        return solution.strip()

    def load_and_embed_solutions(self, json_dir: str):
        """Load solutions from JSON files and create embeddings"""
        # Load all JSON files
        json_files = glob.glob(f"{json_dir}/*.json")

        # Process each solution
        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = json.load(f)

                solution = Solution(
                    title=data['title'],
                    solution=self.process_solution(data),
                    difficulty=data['difficulty'],
                    topics=data['topics'],
                    companies=data['companies']
                )
                self.solutions.append(solution)

        # Create composite texts for embedding
        texts = [
            f"Title: {sol.title}\nDifficulty: {sol.difficulty}\n"
            f"Topics: {sol.topics}\nSolution: {sol.solution}"
            for sol in self.solutions
        ]

        if texts:
            # Generate embeddings
            print("Generating embeddings...")
            embeddings = np.array(self.encoder.encode(
                texts, show_progress_bar=True))

            if embeddings.size > 0:
                # Create and train HNSW index
                print("Creating HNSW index...")
                # 32 is the HNSW parameter (number of neighbors)
                index = faiss.IndexHNSWFlat(self.dimension, 32)

                # Add vectors to index
                index.add(embeddings.astype(np.float32))

                # Save index and metadata
                print("Saving index and metadata...")
                faiss.write_index(index, str(self.index_path))
                with open(self.metadata_path, 'wb') as f:
                    pickle.dump(self.solutions, f)

                print(f"Processed {len(self.solutions)} solutions")
            else:
                print("No embeddings were generated.")
        else:
            print("No solutions found to process.")


if __name__ == "__main__":
    embedder = LeetCodeEmbedder()
    embedder.load_and_embed_solutions("research/leetcode_solutions")
