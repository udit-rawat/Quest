import pickle
import os
from dataclasses import dataclass
from typing import List

# Define the Solution dataclass


@dataclass
class Solution:
    title: str
    solution: str
    difficulty: str
    topics: str
    companies: str


# Function to load metadata
def load_metadata(metadata_path: str) -> List[Solution]:
    """Load metadata from a pickle file."""
    try:
        with open(metadata_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Failed to load metadata: {e}")
        raise


# Function to inspect metadata
def inspect_metadata(metadata: List[Solution], num_entries: int = 5):
    """Inspect the first few entries of the metadata."""
    print("Inspecting metadata...")
    for i, solution in enumerate(metadata[:num_entries]):
        print(f"Entry {i + 1}:")
        print(f"Title: {solution.title}")
        print(f"Difficulty: {solution.difficulty}")
        print(f"Topics: {solution.topics}")
        print(f"Companies: {solution.companies}")
        print("-" * 40)


# Function to validate metadata
def validate_metadata(metadata: List[Solution]):
    """Validate metadata for missing or incorrect data."""
    print("Validating metadata...")
    valid_difficulties = {"Easy", "Medium", "Hard"}

    for solution in metadata:
        # Check difficulty
        if solution.difficulty not in valid_difficulties:
            print(
                f"Invalid difficulty in solution: {solution.title} - {solution.difficulty}")

        # Check topics
        if not solution.topics or solution.topics.lower() == "nan":
            print(f"Missing topics in solution: {solution.title}")

        # Check companies
        if not solution.companies or solution.companies.lower() == "nan":
            print(f"Missing companies in solution: {solution.title}")


# Function to clean metadata
def clean_metadata(metadata: List[Solution]) -> List[Solution]:
    """Clean metadata by fixing missing or incorrect data."""
    print("Cleaning metadata...")
    difficulty_mapping = {
        "easy": "Easy",
        "medium": "Medium",
        "hard": "Hard"
    }

    cleaned_metadata = []
    for solution in metadata:
        # Fix difficulty
        solution.difficulty = difficulty_mapping.get(
            solution.difficulty.lower(), "Unknown")

        # Fix topics
        if not solution.topics or solution.topics.lower() == "nan":
            solution.topics = "Unknown"

        # Fix companies
        if not solution.companies or solution.companies.lower() == "nan":
            solution.companies = "Unknown"

        cleaned_metadata.append(solution)

    return cleaned_metadata


# Function to save cleaned metadata
def save_metadata(metadata: List[Solution], output_path: str):
    """Save cleaned metadata to a pickle file."""
    try:
        with open(output_path, "wb") as f:
            pickle.dump(metadata, f)
        print(f"Cleaned metadata saved to {output_path}")
    except Exception as e:
        print(f"Failed to save metadata: {e}")
        raise


# Main function to verify and clean metadata
def main():
    # Path to the metadata file
    metadata_path = "leetcode_metadata2.pkl"
    cleaned_metadata_path = "cleaned_leetcode_metadata2.pkl"

    # Load metadata
    metadata = load_metadata(metadata_path)

    # Inspect metadata
    inspect_metadata(metadata)

    # Validate metadata
    validate_metadata(metadata)

    # Clean metadata
    cleaned_metadata = clean_metadata(metadata)

    # Save cleaned metadata
    save_metadata(cleaned_metadata, cleaned_metadata_path)

    # Reinitialize the retriever with cleaned metadata (optional)
    # retriever = LeetCodeRetriever(metadata_path=cleaned_metadata_path)


if __name__ == "__main__":
    main()
