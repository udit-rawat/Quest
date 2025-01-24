# Quest

**Quest** is a Retrieval-Augmented Generation (RAG) engine designed to assist with solving and explaining Data Structures and Algorithms (DSA) problems. It leverages a combination of retrieval-based methods and generative models to provide accurate and context-aware solutions to coding problems, explanations of concepts, and metadata about LeetCode-style questions.

---

## Features

- **Exact Matching Retrieval**: Quickly retrieves solutions for exact matches of problem titles from a curated dataset.
- **Context-Aware Generation**: Generates detailed explanations and solutions for general queries using a generative model.
- **Metadata Integration**: Provides metadata such as problem difficulty, related topics, and edge cases for retrieved solutions.
- **Efficient Search**: Uses FAISS and HNSW for fast and accurate similarity search in high-dimensional spaces.
- **Customizable**: Supports different models and configurations for retrieval and generation.
- **Pre-Generated Solutions**: Includes **1800+ solutions in JSON format**, created using the `qwen2.5-coder:1.5b` model on a local machine.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/udit-rawat/Quest.git
   cd Quest
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Unzip the `leetcode_solutions.zip` file to access the solutions in JSON format:
   ```bash
   unzip leetcode_solutions.zip -d src/DSAAssistant/components/
   ```

---

## Setting Up Ollama

To use the generative models (`qwen2.5-coder:1.5b` and `deepseek-r1:1.5b`), you need to install Ollama and pull the required models.

### Install Ollama

- **macOS**:
  Download and install Ollama from the official website:  
  [Ollama for macOS](https://ollama.ai/download/mac)

- **Windows**:
  Download and install Ollama from the official website:  
  [Ollama for Windows](https://ollama.ai/download/windows)

### Pull the Models

Once Ollama is installed, pull the required models using the following commands:

```bash
# Pull qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:1.5b

# Pull deepseek-r1:1.5b
ollama pull deepseek-r1:1.5b
```

---

## Usage

### Running the RAG Engine

1. Start the RAG engine:

   ```bash
   python app.py
   ```

2. Use the engine to query solutions or explanations. Example:
   ```bash
   Query: "Explain the concept of dynamic programming."
   ```

---

## Project Structure

```
Quest/
├── app.py                          # Main application entry point
├── requirements.txt                # List of dependencies
├── .gitignore                      # Files and directories to ignore
├── README.md                       # Project documentation
├── src/
│   ├── DSAAssistant/               # Core components of the RAG engine
│   │   ├── components/
│   │   │   ├── retriever2.py       # LeetCode retriever implementation
│   │   │   ├── rag_engine.py       # RAG engine implementation
│   │   │   ├── leetcode_hnsw.index # Pre-built HNSW index
│   │   │   ├── leetcode_metadata.pkl # Metadata for LeetCode problems
│   │   │   ├── leetcode_solutions.zip # Zipped JSON solutions
│   │   │   └── ...                 # Other components
│   │   └── ...                     # Additional modules
│   └── ...                         # Other source files
└── ...                             # Configuration and other files
```

---

## Configuration

The RAG engine can be configured using the following parameters:

- **Retriever**:

  - `index_path`: Path to the HNSW index file.
  - `metadata_path`: Path to the metadata file.
  - `model_name`: Name of the sentence transformer model for encoding.

- **Generation**:
  - `ollama_url`: URL for the Ollama API (if using a remote model).
  - `model_name`: Name of the generative model.
  - `temperature`: Controls the randomness of the generated output.
  - `top_p`: Controls the diversity of the generated output.

Example configuration:

```python
retriever = LeetCodeRetriever(
    index_path="path/to/leetcode_hnsw.index",
    metadata_path="path/to/leetcode_metadata.pkl",
    model_name="all-MiniLM-L6-v2"
)

rag_engine = RAGEngine(
    retriever=retriever,
    ollama_url="http://localhost:11434/api/generate",
    model_name="qwen2.5-coder:1.5b",
    temperature=0.4,
    top_p=0.9
)
```

---

## Contributing

Contributions are welcome! If you'd like to contribute to **Quest**, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature or fix"
   ```
4. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request and describe your changes.

---

## Acknowledgments

- **FAISS** and **HNSW** for efficient similarity search.
- **Sentence Transformers** for encoding text into embeddings.
- **Ollama** for providing the generative model API.

---

## Contact

For questions or feedback, feel free to reach out:

- **Udit Rawat**
- GitHub: [udit-rawat](https://github.com/udit-rawat)
- Email: [uditrawat1905@gmail.com](mailto:uditrawat1905@gmail.com)

```

```
