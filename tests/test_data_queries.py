from typing import List
from dataclasses import dataclass


@dataclass
class TestQuery:
    query: str
    expected_problems: List[str]
    category: str
    difficulty: str
    challenge_type: str


class LeetCodeTestQueries:
    @staticmethod
    def get_conceptually_ambiguous_queries() -> List[TestQuery]:
        """Queries that are conceptually ambiguous and could match multiple problems"""
        return [
            TestQuery(
                query="find minimum operations to make array beautiful",
                expected_problems=["Minimum Operations to Make Array Beautiful",
                                   "Make Array Zero by Subtracting Equal Amounts"],
                category="array_manipulation",
                difficulty="hard",
                challenge_type="conceptual_ambiguity"
            ),
            TestQuery(
                query="optimize path finding in grid with obstacles using minimum jumps",
                expected_problems=["Jump Game", "Jump Game II",
                                   "Shortest Path in Binary Matrix"],
                category="dynamic_programming",
                difficulty="hard",
                challenge_type="algorithm_overlap"
            ),
            TestQuery(
                query="find maximum sum subarray with alternating positive negative elements",
                expected_problems=["Maximum Subarray",
                                   "Maximum Alternating Subsequence Sum"],
                category="array",
                difficulty="medium",
                challenge_type="requirement_ambiguity"
            ),
            TestQuery(
                query="optimize tree traversal with minimum memory usage without recursion",
                expected_problems=["Binary Tree Inorder Traversal",
                                   "Morris Traversal", "Iterative Tree Traversal"],
                category="trees",
                difficulty="medium",
                challenge_type="implementation_ambiguity"
            ),
            TestQuery(
                query="find shortest cycle in undirected graph with weighted edges",
                expected_problems=["Shortest Cycle in Graph",
                                   "Minimum Spanning Tree", "Network Delay Time"],
                category="graphs",
                difficulty="hard",
                challenge_type="algorithm_complexity"
            )
        ]

    @staticmethod
    def get_semantically_challenging_queries() -> List[TestQuery]:
        """Queries using non-standard terminology or domain-specific language"""
        return [
            TestQuery(
                query="implement memcache with LRU and time-based eviction",
                expected_problems=[
                    "LRU Cache", "Design In-Memory File System", "Time Based Key-Value Store"],
                category="system_design",
                difficulty="hard",
                challenge_type="domain_specific"
            ),
            TestQuery(
                query="optimize matrix chain multiplication using bottom-up approach",
                expected_problems=[
                    "Matrix Chain Multiplication", "Burst Balloons"],
                category="dynamic_programming",
                difficulty="hard",
                challenge_type="technical_terminology"
            ),
            TestQuery(
                query="implement concurrent rate limiter with sliding window",
                expected_problems=[
                    "Rate Limiter", "Sliding Window Maximum", "Design Hit Counter"],
                category="system_design",
                difficulty="hard",
                challenge_type="cross_domain"
            ),
            TestQuery(
                query="find strongly connected components in directed graph without Tarjan",
                expected_problems=[
                    "Critical Connections in a Network", "Strongly Connected Components"],
                category="graphs",
                difficulty="hard",
                challenge_type="algorithm_constraint"
            ),
            TestQuery(
                query="optimize database index structure using B+ tree with minimum memory",
                expected_problems=["Design In-Memory File System",
                                   "Design Search Autocomplete System"],
                category="system_design",
                difficulty="hard",
                challenge_type="system_specific"
            )
        ]

    @staticmethod
    def get_edge_case_queries() -> List[TestQuery]:
        """Queries testing edge cases and boundary conditions"""
        return [
            TestQuery(
                query="handle concurrent modifications in binary search tree with duplicates",
                expected_problems=[
                    "Binary Search Tree Iterator", "Serialize and Deserialize BST"],
                category="trees",
                difficulty="hard",
                challenge_type="concurrency_edge"
            ),
            TestQuery(
                query="optimize space complexity for computing large fibonacci numbers with modulo",
                expected_problems=["Fibonacci Number",
                                   "Matrix Exponentiation"],
                category="math",
                difficulty="medium",
                challenge_type="optimization_edge"
            ),
            TestQuery(
                query="handle overflow in binary tree with maximum path sum allowing negative values",
                expected_problems=[
                    "Binary Tree Maximum Path Sum", "Path Sum II"],
                category="trees",
                difficulty="hard",
                challenge_type="numeric_edge"
            ),
            TestQuery(
                query="find minimum cuts in flow network with negative capacity edges",
                expected_problems=["Network Flow",
                                   "Minimum Cut", "Maximum Flow"],
                category="graphs",
                difficulty="hard",
                challenge_type="constraint_edge"
            ),
            TestQuery(
                query="optimize palindrome checking for very large strings with unicode characters",
                expected_problems=["Valid Palindrome",
                                   "Longest Palindromic Substring"],
                category="strings",
                difficulty="medium",
                challenge_type="input_edge"
            )
        ]


if __name__ == "__main__":
    test_queries = LeetCodeTestQueries()
    print(test_queries.get_conceptually_ambiguous_queries())
    print(test_queries.get_semantically_challenging_queries())
    print(test_queries.get_edge_case_queries())
