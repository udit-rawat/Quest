import json
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class LeetCodeInteractiveRAG:
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.problems: Dict[str, dict] = {}
        self.index = None
        self.embeddings = None

    def load_problems(self, json_files: List[str]):
        for file in json_files:
            with open(file) as f:
                problem = json.load(f)
                self.problems[problem['title']] = problem

        # Create composite texts for search
        texts = [
            f"{p['title']} {p['description']} {p['topics']}"
            for p in self.problems.values()
        ]

        # Create FAISS index
        embeddings = self.encoder.encode(texts)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexHNSWFlat(dimension, 32)
        self.index.add(embeddings)
        self.embeddings = embeddings

    def find_similar_problem(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(query_vector, k)

        return [
            (list(self.problems.keys())[idx], float(dist))
            for idx, dist in zip(indices[0], distances[0])
        ]

    def get_problem_info(self, title: str, aspect: str) -> str:
        problem = self.problems.get(title)
        if not problem:
            return "Problem not found"

        if aspect == "intuition":
            # Extract intuition from solution comments
            solution = problem['solution']
            if "Approach:" in solution:
                return solution.split("Approach:")[1].split("\n")[0]
            return "Let's break this down step by step..."

        elif aspect == "similar":
            return problem.get('similar_questions', 'No similar questions found')

        elif aspect == "companies":
            return problem.get('companies', 'No company information available')

        elif aspect == "hints":
            # Extract key points without giving away solution
            topics = problem.get('topics', '').split(',')
            return f"This is a {topics[0]} problem. Think about using {topics[-1]}"

        return problem.get(aspect, "Information not available")

    def get_progressive_hints(self, title: str) -> List[str]:
        problem = self.problems.get(title)
        if not problem:
            return []

        # Extract solution approach steps
        solution = problem['solution']
        if "Approach:" in solution:
            steps = solution.split("Approach:")[1].split("\n")
            return [step.strip() for step in steps if step.strip()]
        return ["Think about the problem constraints",
                "Consider edge cases",
                "Try breaking it into smaller steps"]
