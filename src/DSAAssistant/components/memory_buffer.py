# conversation_history.py
from typing import List, Dict


class ConversationHistory:
    def __init__(self, max_history: int = 5):
        """
        Initialize the conversation history with a maximum limit.
        :param max_history: Maximum number of queries to retain in history.
        """
        self.max_history = max_history
        # Stores queries and their context
        self.history: List[Dict[str, str]] = []

    def add_query(self, query: str, response: str):
        """
        Add a new query and response to the history.
        :param query: The user's query.
        :param response: The system's response.
        """
        self.history.append({"query": query, "response": response})
        if len(self.history) > self.max_history:
            # Remove the oldest query if history exceeds the limit
            self.history.pop(0)

    def get_context(self) -> str:
        """
        Generate a context string from the conversation history.
        :return: A formatted context string.
        """
        context = ""
        for entry in self.history:
            context += f"User: {entry['query']}\nSystem: {entry['response']}\n"
        return context.strip()

    def clear(self):
        """Clear the conversation history."""
        self.history = []
