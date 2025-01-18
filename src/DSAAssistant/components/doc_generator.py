# Used for documentation generation via Qwen2.5-coder model
import json
import time
import requests
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


class LeetCodeSolutionGenerator:
    def __init__(self, model_name="qwen2.5-coder:1.5b", output_dir="./leetcode_solutions"):
        """Initialize the solution generator."""
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    # meta prompt for solution generation.
    def create_prompt(self, problem: pd.Series) -> str:
        """Create a detailed prompt for code generation."""
        prompt = f"""
Generate the most optimal Python solution for this LeetCode problem.

Problem:
Title: {problem['title']}
Difficulty: {problem['difficulty']}
Description: {problem['description']}
Topics: {problem['related_topics']}

Requirements:
1. Most optimal solution (best time and space complexity)
2. Well-documented with complexity analysis
3. Include test cases
4. Handle all edge cases

Write the solution in this format:
```python
def solution(params):
    \"\"\"
    {problem['title']}
    
    Time: O(?)  # Specify exact complexity
    Space: O(?) # Specify exact complexity
    
    Approach:
    1. Step-by-step explanation
    2. Why this approach is optimal
    3. How edge cases are handled
    \"\"\"
    # Implementation
    pass

# Test cases
if __name__ == "__main__":
    assert solution(input1) == expected1
    assert solution(input2) == expected2
```"""
        return prompt

    def generate_solution(self, prompt: str) -> str:
        """Generate solution using the model."""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "stream": False
                }
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return None
        except Exception as e:
            print(f"Error generating solution: {e}")
            return None

    def save_solution(self, problem: pd.Series, solution: str):
        """Save the solution to a separate JSON file."""
        file_name = f"{problem['title'].lower().replace(' ', '_').replace('/', '_')}.json"
        file_path = self.output_dir / file_name

        data = {
            "title": problem['title'],
            "difficulty": problem['difficulty'],
            "description": problem['description'],
            "solution": solution,
            "topics": problem['related_topics'],
            "companies": problem.get('companies', []),
            "url": problem['url'],
            "similar_questions": problem.get('similar_questions', []),
            "metadata": {
                "source": "leetcode",
                "generated_date": time.strftime("%Y-%m-%d"),
                "model": self.model_name
            }
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved solution to {file_path}")

    def process_problems(self, df: pd.DataFrame, start: int, end: int):
        """Process a range of problems and generate solutions."""
        for idx, problem in tqdm(df.iloc[start:end].iterrows(), total=end-start):
            try:
                # Generate solution
                prompt = self.create_prompt(problem)
                solution = self.generate_solution(prompt)
                if solution:
                    self.save_solution(problem, solution)
                time.sleep(0.5)  # Delay to avoid overloading the server
            except Exception as e:
                print(f"Error processing {problem['title']}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(
        description="Generate LeetCode solutions using a local model.")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Path to the CSV dataset.")
    parser.add_argument('--start', type=int, default=0,
                        help="Start index for processing problems.")
    parser.add_argument('--end', type=int, default=None,
                        help="End index for processing problems.")
    parser.add_argument('--output_dir', type=str,
                        default="./leetcode_solutions", help="Directory to save solutions.")

    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.dataset)

    # Initialize solution generator
    generator = LeetCodeSolutionGenerator(output_dir=args.output_dir)

    # Process problems
    generator.process_problems(
        df, args.start, args.end if args.end else len(df))


if __name__ == "__main__":
    main()
