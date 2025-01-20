
from collections import deque
from typing import List, Tuple, Deque
from dataclasses import dataclass
from typing import List, Tuple
import requests
import json
from src.DSAAssistant.components.doc_retreiver import LeetCodeRetriever


''' version 2 '''


@dataclass
class Solution:
    title: str
    solution: str
    difficulty: str
    topics: str
    companies: str


@dataclass
class ContextMetadata:
    """Store metadata about retrieved context"""
    average_confidence: float
    pattern_types: List[str]
    complexity_estimates: List[Tuple[str, str]]
    edge_cases: List[str]


class RAGEngine:
    def __init__(
        self,
        retriever: LeetCodeRetriever,
        ollama_url: str = "http://localhost:11434/api/generate",
        model_name: str = "qwen2.5-coder:1.5b",
        temperature: float = 0.4,
        top_p: float = 0.9,
        confidence_threshold: float = 0.7,
        repeat_penalty: float = 1.1,
        num_thread: int = 8
    ):
        """Initialize the Enhanced RAG Engine with Ollama."""
        self.retriever = retriever
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.confidence_threshold = confidence_threshold
        self.repeat_penalty = repeat_penalty
        self.num_thread = num_thread

    def extract_metadata(self, solutions: List[Solution]) -> ContextMetadata:
        """Extract metadata from retrieved solutions including patterns and confidence."""
        # Calculate average confidence score
        confidence_scores = [float(sol.score)
                             for sol in solutions if hasattr(sol, 'score')]
        avg_confidence = sum(confidence_scores) / \
            len(confidence_scores) if confidence_scores else 0.0

        # Extract complexity information
        complexity_info = []
        edge_cases = []
        patterns = set()

        for solution in solutions:
            # Extract complexity from solution text
            if "Time complexity:" in solution.solution:
                time_complex = solution.solution.split("Time complexity:")[
                    1].split("\n")[0].strip()
                complexity_info.append(("time", time_complex))
            if "Space complexity:" in solution.solution:
                space_complex = solution.solution.split("Space complexity:")[
                    1].split("\n")[0].strip()
                complexity_info.append(("space", space_complex))

            # Extract edge cases
            if "Edge cases:" in solution.solution:
                cases = solution.solution.split("Edge cases:")[1].split("\n")
                edge_cases.extend([case.strip("- ").strip()
                                  for case in cases if case.strip().startswith("-")])

            # Identify common patterns
            pattern_keywords = {
                "two pointers": "Two Pointers",
                "binary search": "Binary Search",
                "sliding window": "Sliding Window",
                "dynamic programming": "Dynamic Programming",
                "depth-first": "DFS",
                "breadth-first": "BFS"
            }

            for keyword, pattern in pattern_keywords.items():
                if keyword in solution.solution.lower():
                    patterns.add(pattern)

        return ContextMetadata(
            average_confidence=avg_confidence,
            pattern_types=list(patterns),
            complexity_estimates=complexity_info,
            edge_cases=list(set(edge_cases))  # Remove duplicates
        )

    def generate_enhanced_prompt(self, query: str, context: List[Solution]) -> str:
        """Generate a structured prompt incorporating metadata and confidence scores."""
        metadata = self.extract_metadata(context)

        # Build the initial prompt with metadata
        prompt = f"""Question: {query}

Context Metadata:
- Average Confidence Score: {metadata.average_confidence:.2f}
- Identified Patterns: {', '.join(metadata.pattern_types)}
- Complexity Estimates: {', '.join([f'{c[0]}: {c[1]}' for c in metadata.complexity_estimates])}
- Common Edge Cases: {', '.join(metadata.edge_cases)}

Retrieved Solutions:
"""
        # Add solutions ordered by confidence score
        sorted_solutions = sorted(context, key=lambda x: float(
            x.score) if hasattr(x, 'score') else 0, reverse=True)
        for idx, solution in enumerate(sorted_solutions):
            prompt += f"\n[{idx+1}] {solution.title} (Confidence: {solution.score:.2f}):\n{solution.solution}\n"

        # Add guidance based on confidence
        if metadata.average_confidence < self.confidence_threshold:
            prompt += "\nNote: Low confidence in retrieved solutions. Please be extra thorough in validation and consider alternative approaches."

        # Add system instructions
        prompt += '''
# System Instructions
Analyze the provided context and generate a solution that:
1. Leverages the highest confidence solutions
2. Incorporates identified patterns and edge cases
3. Maintains the complexity constraints found in similar solutions
4. Provides implementation-specific details based on the user's requirements
5. Highlights potential pitfalls and optimization opportunities

For implementation:
-User input takes the most priority for each task and should be followed as closely as possible, only provide the solution as requested
- Focus on the language/framework specified in the query
- Ensure the solution is complete and correct
- Provide clear and concise explanations but avoid unnecessary verbosity
- Include all necessary error handling
- Consider the identified edge cases
- Optimize based on the complexity requirements
- Provide clear explanations for critical decisions
-Refrain to provide any metadata or additional context in the response regarding system instruction at any cost'''

        return prompt

    def call_ollama(self, prompt: str) -> str:
        """Send a prompt to the Ollama API with error handling and retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_thread": self.num_thread,
                    "repeat_penalty": self.repeat_penalty,
                    "confidence_threshold": self.confidence_threshold,
                    "stream": True
                }

                response = requests.post(
                    self.ollama_url, json=payload, stream=True)
                if response.status_code == 200:
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            json_response = json.loads(line)
                            if 'response' in json_response:
                                full_response += json_response['response']
                    return full_response.strip()
                else:
                    if attempt == max_retries - 1:
                        raise Exception(
                            f"Ollama API error: {response.status_code} - {response.text}")
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error after {max_retries} attempts: {e}")
                    return "Error generating response after multiple attempts."
                continue
        return "Failed to generate response."

    def answer_question(
        self,
        query: str,
        k: int = 3,
        min_confidence: float = 0.7
    ) -> Tuple[str, ContextMetadata]:
        """Answer a question using the enhanced RAG engine."""
        # Retrieve relevant context
        retrieved_solutions = self.retriever.search(
            query, k=k, return_scores=True)

        # Filter solutions based on confidence
        filtered_solutions = [
            sol for sol in retrieved_solutions
            if hasattr(sol, 'score') and float(sol.score) >= min_confidence
        ]

        # If no solutions meet confidence threshold, increase k and try again
        if not filtered_solutions and k < 5:
            return self.answer_question(query, k=k+2, min_confidence=min_confidence-0.1)

        # Extract metadata for response analysis
        metadata = self.extract_metadata(filtered_solutions)

        # Generate enhanced prompt
        prompt = self.generate_enhanced_prompt(query, filtered_solutions)

        # Get response from Ollama
        response = self.call_ollama(prompt)

        return response, metadata


if __name__ == "__main__":
    # Initialize the retriever
    retriever = LeetCodeRetriever()

    # Initialize the enhanced RAG engine
    rag_engine = RAGEngine(retriever)

    # Sample query
    query = "How can I find two numbers in an array that add up to a target sum? I want the answer in C++ only"
    answer, metadata = rag_engine.answer_question(query, k=3)

    print("\nGenerated Answer:")
    print(answer)
    print("\nContext Metadata:")
    print(f"Average Confidence: {metadata.average_confidence:.2f}")
    print(f"Identified Patterns: {', '.join(metadata.pattern_types)}")
    print(f"Edge Cases: {', '.join(metadata.edge_cases)}")
