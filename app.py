from flask import Flask, request, jsonify, render_template, Response
from collections import deque
from rag_engine import RAGEngine, LeetCodeRetriever
from dataclasses import dataclass
import json
import time

app = Flask(__name__)


@dataclass
class Solution:
    title: str
    solution: str
    difficulty: str
    topics: str
    companies: str


# Initialize retriever and RAG engine
retriever = LeetCodeRetriever()
rag_engine = RAGEngine(retriever)


class MemoryBuffer:
    def __init__(self, max_length=10):
        self.buffer = deque(maxlen=max_length)

    def add(self, user_input, model_response):
        self.buffer.append({
            "user_input": user_input,
            "model_response": model_response
        })

    def get_context(self):
        return "\n".join([
            f"User: {item['user_input']}\nModel: {item['model_response']}"
            for item in self.buffer
        ])

    def clear(self):
        self.buffer.clear()


memory = MemoryBuffer()


def stream_tokens(text):
    """Simulate token-by-token streaming."""
    words = text.split()
    for i, word in enumerate(words):
        yield json.dumps({
            "type": "token",
            "content": " ".join(words[:i+1])
        }) + "\n"
        time.sleep(0.05)  # Adjust timing as needed


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream", methods=["POST"])
def stream():
    data = request.json
    user_input = data.get("user_input")

    if not user_input:
        return jsonify({"error": "User input is required."}), 400

    def generate():
        # Get context from memory
        context = memory.get_context()

        # Generate answer using the RAG engine
        query_with_context = f"{context}\nUser: {user_input}"
        answer, metadata = rag_engine.answer_question(query_with_context)

        # Stream the response token by token
        for token in stream_tokens(answer):
            yield token

        # Send metadata at the end
        yield json.dumps({
            "type": "metadata",
            "metadata": metadata.__dict__
        }) + "\n"

        # Add to memory after streaming
        memory.add(user_input, answer)

    return Response(generate(), mimetype='text/event-stream')


@app.route("/history", methods=["GET"])
def history():
    return jsonify(list(memory.buffer))


@app.route("/clear_history", methods=["POST"])
def clear_history():
    memory.clear()
    return jsonify({"message": "History cleared successfully."})


if __name__ == "__main__":
    app.run(debug=True)  # --version2
