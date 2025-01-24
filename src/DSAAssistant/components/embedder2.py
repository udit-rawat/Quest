import json
import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
import glob
import pickle
from pathlib import Path
from dataclasses import dataclass
import logging


# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


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
        index_path: str = "leetcode_hnsw2.index",
        metadata_path: str = "leetcode_metadata2.pkl"
    ):
        self.encoder = SentenceTransformer(model_name)
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.solutions: List[Solution] = []
        self.dimension = 384  # MiniLM-L6-v2 dimension

    def validate_metadata(self, json_data):
        """Validate metadata fields in JSON data."""
        required_fields = ["title", "difficulty", "topics",
                           "companies", "url", "similar_questions"]
        for field in required_fields:
            if field not in json_data:
                logger.warning(
                    f"Missing required field: {field} in {json_data.get('title', 'unknown')}")

    def clean_metadata(self, json_data):
        """Clean and standardize metadata fields."""
        # Ensure companies is a string
        companies = json_data.get("companies")
        if companies is None:  # Handle missing field
            companies = "N/A"
        elif isinstance(companies, (int, float)):  # Handle numbers
            companies = str(companies)
        elif not isinstance(companies, str):  # Handle invalid types
            companies = "N/A"
        json_data["companies"] = companies.strip()

        # Clean other fields
        json_data["difficulty"] = str(json_data.get(
            "difficulty", "N/A")).strip().capitalize()
        json_data["topics"] = str(json_data.get("topics", "N/A")).strip()
        return json_data

    def process_solution(self, json_data: dict) -> str:
        """Extract and clean solution text from JSON."""
        solution = json_data.get('solution', '')
        # Remove markdown code blocks
        solution = solution.replace('```python', '').replace('```', '')
        # Extract approach and implementation details
        if 'Approach:' in solution:
            solution = solution.split('Approach:')[1]
        return solution.strip()

    def load_and_embed_solutions(self, json_dir: str):
        """Load solutions from JSON files and create embeddings."""
        # Load all JSON files
        json_files = glob.glob(f"{json_dir}/*.json")

        # Process each solution
        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.validate_metadata(data)
                data = self.clean_metadata(data)

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
            f"Topics: {sol.topics}\nCompanies: {sol.companies}\n"
            f"Solution: {sol.solution}"
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
    embedder.load_and_embed_solutions("leetcode_solutions")
