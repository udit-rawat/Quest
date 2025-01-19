from flask import Flask, request, jsonify, render_template
from collections import deque
# Your RAGEngine import
from rag_engine import RAGEngine, LeetCodeRetriever
from dataclasses import dataclass
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

# Memory buffer


class MemoryBuffer:
    def __init__(self, max_length=10):
        self.buffer = deque(maxlen=max_length)

    def add(self, user_input, model_response):
        self.buffer.append(
            {"user_input": user_input, "model_response": model_response})

    def get_context(self):
        return "\n".join(
            [f"User: {item['user_input']}\nModel: {item['model_response']}" for item in self.buffer]
        )

    def clear(self):
        self.buffer.clear()


memory = MemoryBuffer()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_input = data.get("user_input")

    if not user_input:
        return jsonify({"error": "User input is required."}), 400

    # Get context from memory
    context = memory.get_context()

    # Generate answer using the RAG engine
    query_with_context = f"{context}\nUser: {user_input}"
    answer, metadata = rag_engine.answer_question(query_with_context)

    # Add to memory
    memory.add(user_input, answer)

    return jsonify({
        "user_input": user_input,
        "response": answer,
        "metadata": metadata.__dict__,
    })


@app.route("/history", methods=["GET"])
def history():
    return jsonify(list(memory.buffer))


@app.route("/clear_history", methods=["POST"])
def clear_history():
    memory.clear()
    return jsonify({"message": "History cleared successfully."})


if __name__ == "__main__":
    app.run(debug=True)  # --version1
