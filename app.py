from flask import Flask, render_template, request, Response, stream_with_context
from src.DSAAssistant.components.doc_retriever import LeetCodeRetriever, Solution
from rag_engine import RAGEngine

app = Flask(__name__)
retriever = LeetCodeRetriever()
rag_engine = RAGEngine(retriever)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query')
    mode = data.get('mode', 'general')

    if not query:
        return Response('Query is required', status=400)

    def generate():
        try:
            # Set the mode
            rag_engine.set_mode(mode)

            # Get response from RAG engine
            response, _ = rag_engine.answer_question(query)

            # Stream the response
            yield response

        except Exception as e:
            app.logger.error(f"Error processing query: {str(e)}")
            yield "An error occurred while processing your request."

    return Response(stream_with_context(generate()), mimetype='text/plain')


if __name__ == '__main__':
    app.run(debug=True)
