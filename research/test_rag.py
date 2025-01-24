import os
from mvm import LeetCodeInteractiveRAG

# Initialize the RAG class
rag = LeetCodeInteractiveRAG()

# Path to the directory containing JSON files
json_dir = 'research/leetcode_solutions'

# List all JSON files in the directory
json_files = [os.path.join(json_dir, file)
              for file in os.listdir(json_dir) if file.endswith('.json')]

# Load the problems from the JSON files
rag.load_problems(json_files)

# Allow the user to input a query
query = input("Enter your query: ")

# Test finding similar problems
similar_problems = rag.find_similar_problem(query)
print(f"Similar problems to '{query}':")
for title, distance in similar_problems:
    print(f"Title: {title}, Distance: {distance}")

# Test getting problem information
if similar_problems:
    # Get the title of the first similar problem
    title = similar_problems[0][0]
    aspect = "intuition"
    info = rag.get_problem_info(title, aspect)
    print(f"\nIntuition for '{title}': {info}")

    # Test getting progressive hints
    hints = rag.get_progressive_hints(title)
    print(f"\nProgressive hints for '{title}':")
    for hint in hints:
        print(hint)
else:
    print("No similar problems found.")
