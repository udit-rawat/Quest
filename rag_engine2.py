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
        reasoning_model: str = "deepseek-r1:1.5b",  # Reasoning model
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
        logger.info("RAG Engine initialized successfully.")

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
        min_confidence: float = 0.5
    ) -> str:
        """Answer a question using the enhanced RAG engine."""
        try:
            # Reset the stop flag before starting a new generation
            self.reset()

            # Add the query to the conversation history
            context = self.conversation_history.get_context()
            prompt = self.generate_enhanced_prompt(query, context)

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
    retriever = LeetCodeRetriever()
    rag_engine = RAGEngine(retriever)

    # Test multi-turn conversation
    queries = [
        "Explain the concept of dynamic programming.",
        "How can I optimize a recursive Fibonacci function using memoization?",
        "What is the time complexity of the memoized Fibonacci function?",
        "Can you provide an example of dynamic programming in real life?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        answer = rag_engine.answer_question(query, k=3)
        print("\nGenerated Answer:")
        print(answer)

    # Simulate stopping the generation after 5 seconds
    time.sleep(10)
    rag_engine.stop()
    print("\nGeneration stopped by user.")
