from flask import Flask, render_template, request, Response, jsonify
from src.DSAAssistant.components.retriever2 import LeetCodeRetriever, Solution
# Ensure this import points to your RAGEngine class
from rag_engine3 import RAGEngine
import logging
import time

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG Engine
retriever = LeetCodeRetriever()
rag_engine = RAGEngine(retriever, max_history=3)  # Initialize with max_history


@app.route('/')
def index():
    """Render the main index page."""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """
    Handle search requests from the frontend.
    Returns the entire response at once (no streaming).
    """
    # Get JSON data from the request
    data = request.get_json()
    query = data.get('query')
    mode = data.get('mode', 'general')  # Default mode is 'general'

    # Validate the query
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Log the start time
        start_time = time.time()

        # Set the mode (general or reasoning)
        rag_engine.set_mode(mode)
        logger.info(f"Mode set to: {mode}")

        # Get response from RAG engine
        response = rag_engine.answer_question(query)

        # Log the time taken
        logger.info(
            f"Response generated in {time.time() - start_time:.2f} seconds")

        # Return the response as JSON
        return jsonify({"response": response})

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500


@app.route('/stop', methods=['POST'])
def stop():
    """Stop the ongoing generation process."""
    rag_engine.stop()
    return jsonify({"message": "Streaming stopped"}), 200


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear the conversation history."""
    rag_engine.conversation_history.clear()
    return jsonify({"message": "Conversation history cleared"}), 200


@app.route('/get_history', methods=['GET'])
def get_history():
    """Get the current conversation history."""
    history = rag_engine.conversation_history.get_context()
    return jsonify({"history": history})


@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Set the mode (general or reasoning)."""
    data = request.get_json()
    mode = data.get('mode')

    if mode not in ["general", "reasoning"]:
        return jsonify({"error": "Mode must be 'general' or 'reasoning'."}), 400

    rag_engine.set_mode(mode)
    return jsonify({"message": f"Mode set to: {mode}"}), 200


if __name__ == '__main__':
    app.run(debug=False)
