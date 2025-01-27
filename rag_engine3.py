# rag_engine.py
from src.DSAAssistant.components.retriever2 import LeetCodeRetriever, Solution
# Import the ConversationHistory class
from src.DSAAssistant.components.memory_buffer import ConversationHistory
from src.DSAAssistant.components.prompt_temp import PromptTemplates
from typing import List
import requests
import json
import logging
import time
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(
        self,
        retriever: LeetCodeRetriever,
        ollama_url: str = "http://localhost:11434/api/generate",
        model_name: str = "qwen2.5-coder:1.5b",  # Default model
        reasoning_model: str = "deepseek-r1:7b",  # Reasoning model
        mode: str = "general",  # Default mode
        temperature: float = 0.4,
        top_p: float = 0.9,
        confidence_threshold: float = 0.7,
        repeat_penalty: float = 1.1,
        num_thread: int = 8,
        max_history: int = 3  # Add max_history parameter
    ):
        """Initialize the Enhanced RAG Engine with Ollama."""
        self.retriever = retriever
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.reasoning_model = reasoning_model
        self.mode = mode
        self.temperature = temperature
        self.top_p = top_p
        self.confidence_threshold = confidence_threshold
        self.repeat_penalty = repeat_penalty
        self.num_thread = num_thread
        self.stop_generation = False
        self.conversation_history = ConversationHistory(
            max_history)  # Initialize conversation history

        # Build a hash map for exact match search
        self.exact_match_map = self._build_exact_match_map()
        logger.info("RAG Engine initialized successfully.")

    def _build_exact_match_map(self) -> dict:
        """
        Build a hash map for exact match search.
        :return: A dictionary where keys are normalized titles and values are Solution objects.
        """
        exact_match_map = {}
        for solution in self.retriever.solutions:
            normalized_title = self._normalize_title(solution.title)
            exact_match_map[normalized_title] = solution
        return exact_match_map

    def _normalize_title(self, title: str) -> str:
        """
        Normalize a title for exact match search.
        :param title: The title to normalize.
        :return: The normalized title.
        """
        # Convert to lowercase and remove special characters
        return title.strip().lower()

    def set_mode(self, mode: str):
        """Set the mode (general or reasoning)."""
        if mode not in ["general", "reasoning"]:
            raise ValueError("Mode must be 'general' or 'reasoning'.")
        self.mode = mode
        logger.info(f"Mode set to: {mode}")

    def stop(self):
        """Stop the ongoing generation process."""
        self.stop_generation = True
        logger.info("Generation process stopped.")

    def reset(self):
        """Reset the stop flag to allow new generations."""
        self.stop_generation = False
        logger.info("Generation process reset.")

    def generate_enhanced_prompt(self, query: str, context: List[Solution]) -> str:
        """Generate a structured prompt incorporating context."""
        if self.mode == "reasoning":
            return PromptTemplates.reasoning_prompt(query, context)
        else:
            return PromptTemplates.general_prompt(query, context)

    def call_ollama(self, prompt: str) -> str:
        """Send a prompt to the Ollama API with error handling."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Select model based on mode
                model = self.reasoning_model if self.mode == "reasoning" else self.model_name

                payload = {
                    "model": model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_thread": self.num_thread,
                    "repeat_penalty": self.repeat_penalty,
                    "stream": True
                }
                response = requests.post(
                    self.ollama_url, json=payload, stream=True, timeout=30)
                if response.status_code == 200:
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            if self.stop_generation:
                                logger.info("Generation stopped by user.")
                                return full_response.strip()  # Return partial response
                            json_response = json.loads(line)
                            if 'response' in json_response:
                                full_response += json_response['response']
                    return full_response.strip()
                else:
                    logger.warning(
                        f"Ollama API error (attempt {attempt + 1}): {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(
                    f"Ollama API call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return "Error generating response after multiple attempts."
        return "Failed to generate response."

    def filter_reasoning_response(self, response: str) -> str:
        """Filter out the 'think' part from Deepseek's reasoning response."""
        if "<think>" in response and "</think>" in response:
            # Split the response into parts before and after the <think> block
            parts = response.split("</think>")
            if len(parts) > 1:
                # Return the part after </think>
                return parts[1].strip()
        return response  # Return the original response if no <think> block is found

    def answer_question(
        self,
        query: str,
        k: int = 5,
        min_confidence: float = 0.6
    ) -> str:
        """Answer a question using the enhanced RAG engine."""
        try:
            # Reset the stop flag before starting a new generation
            self.reset()

            # Normalize the query for exact matching
            normalized_query = self._normalize_title(query)

            # Check for exact match in the hash map
            if normalized_query in self.exact_match_map:
                exact_match_solution = self.exact_match_map[normalized_query]
                logger.info("Exact match found. Returning solution directly.")
                return f"Exact Match Solution:\n{exact_match_solution.solution}"

            # Retrieve relevant context
            retrieved_solutions = self.retriever.search(
                query, k=k, return_scores=True)
            filtered_solutions = [sol for sol in retrieved_solutions if hasattr(
                sol, 'score') and float(sol.score) >= min_confidence]

            # Fallback if no solutions meet confidence threshold
            if not filtered_solutions and k < 5:
                return self.answer_question(query, k=k + 2, min_confidence=min_confidence - 0.1)

            # Generate enhanced prompt
            prompt = self.generate_enhanced_prompt(query, filtered_solutions)

            # Get response from Ollama
            response = self.call_ollama(prompt)

            # Add the response to the conversation history
            self.conversation_history.add_query(query, response)

            # Filter response if in reasoning mode
            if self.mode == "reasoning":
                response = self.filter_reasoning_response(response)

            return f"Generated Solution:\n{response}"
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return "An error occurred while generating the response."


if __name__ == "__main__":
    # Initialize the RAG engine with the improved reasoning prompt
    retriever = LeetCodeRetriever()
    rag_engine = RAGEngine(retriever)
    rag_engine.set_mode("general")

    # List of queries to test the reasoning mode
    queries = [
        # Conceptual Understanding
        "Can you explain the 'Two Sum' problem in simple terms? What is the goal of the problem?",
        "Why is it important to ensure that the same element is not used twice in the 'Two Sum' problem?",
        "What are the key steps to solve the 'Two Sum' problem?",
        "Why is a hash map (dictionary) a good data structure for solving this problem?",
        "What is the time complexity of the brute-force approach for 'Two Sum'? Can you explain why?",
        "How does using a hash map improve the time complexity? What is the space complexity of this approach?",
    ]

    # Test the reasoning mode with all queries
    for query in queries:
        print(f"\nQuery: {query}")
        answer = rag_engine.answer_question(query, k=3)
        print("\nGenerated Answer:")
        print(answer)

        # Print the current conversation history after each query
        print("\nConversation History:")
        print(rag_engine.conversation_history.get_context())

    # Simulate stopping the generation after all queries are processed
    rag_engine.stop()
    print("\nGeneration stopped by user.")

    # Clear the conversation history and verify it's empty
    rag_engine.conversation_history.clear()
    print("\nConversation History After Clearing:")
    print(rag_engine.conversation_history.get_context())
