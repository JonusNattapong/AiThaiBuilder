import os
import sys
import json
import argparse
import random
from tqdm import tqdm
import csv
import uuid

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from utils.deepseek_utils import generate_with_deepseek, get_deepseek_api_key

from typing import Optional

def generate_tot_prompt(problem: str, current_path: Optional[list] = None, depth: int = 0, max_depth: int = 2):
    """
    Generates prompts for Tree of Thoughts (ToT) style reasoning.
    This is a simplified iterative prompt.
    """
    system_prompt = (
        "You are an AI assistant that solves problems by exploring a tree of thoughts. "
        "At each step, generate a few distinct, viable thoughts or next steps. "
        "Then, for each thought, briefly evaluate its promise or provide a critique."
    )

    path_str = " -> ".join(current_path) if current_path else "Initial problem"
    
    user_prompt = (
        f"Problem: {problem}\n"
        f"Current reasoning path: {path_str}\n\n"
    )

    if depth < max_depth:
        user_prompt += (
            "Please generate 2-3 distinct and promising next thoughts or steps to continue solving the problem from the current path. "
            "For each thought, also provide a brief evaluation or critique of its potential.\n"
            "Format each as:\n"
            "Thought X: <Your thought/next step>\n"
            "Evaluation X: <Brief evaluation/critique of Thought X>\n"
        )
    else:
        user_prompt += (
            "You have reached the maximum exploration depth for this branch. "
            "Based on the current path, provide a concluding thought or a potential solution fragment.\n"
            "Conclusion/Solution Fragment: <your conclusion for this path>"
        )
    return system_prompt, user_prompt

def process_tot_llm_response(llm_response: str, depth: int, max_depth: int):
    """
    Parses the LLM response to extract thoughts and evaluations.
    Returns a list of (thought, evaluation) tuples or a final conclusion.
    This is a very basic parser.
    """
    # TODO: Implement more robust parsing for "Thought X:" and "Evaluation X:"
    # or "Conclusion/Solution Fragment:"
    # This is highly dependent on how consistently the LLM follows the format.
    
    thoughts_and_evaluations = []
    if depth < max_depth:
        # Crude parsing attempt
        lines = llm_response.split('\n')
        current_thought = None
        for line in lines:
            if line.startswith("Thought "): # e.g., "Thought 1:", "Thought A:"
                if current_thought: # Save previous thought before starting a new one
                    thoughts_and_evaluations.append(current_thought)
                current_thought = {"thought": line.split(":", 1)[1].strip() if ":" in line else line.strip(), "evaluation": "N/A"}
            elif line.startswith("Evaluation ") and current_thought: # e.g., "Evaluation 1:"
                 current_thought["evaluation"] = line.split(":", 1)[1].strip() if ":" in line else line.strip()
        if current_thought: # Add the last parsed thought
            thoughts_and_evaluations.append(current_thought)
        
        if not thoughts_and_evaluations: # Fallback if parsing fails
            return [{"thought": "Fallback: Could not parse distinct thoughts.", "evaluation": llm_response}]
        return thoughts_and_evaluations

    else: # At max depth, expect a conclusion
        conclusion_marker = "Conclusion/Solution Fragment:"
        if conclusion_marker in llm_response:
            return [{"conclusion": llm_response.split(conclusion_marker, 1)[1].strip()}]
        return [{"conclusion": "Fallback: Could not parse conclusion: " + llm_response}]


def explore_thought_tree_recursively(problem: str, api_key: str, current_path: list, current_depth: int, max_depth: int, all_paths: list):
    """
    Recursively explores the thought tree.
    A very simplified ToT simulation.
    """
    if current_depth > max_depth:
        all_paths.append({"path": list(current_path), "status": "max_depth_reached", "final_node_output": current_path[-1] if current_path else "N/A"})
        return

    system_prompt, user_prompt = generate_tot_prompt(problem, current_path, current_depth, max_depth)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    llm_response = generate_with_deepseek(messages, api_key, max_tokens=1000)

    if not llm_response or llm_response.startswith("Error:"):
        print(f"Warning: LLM error at depth {current_depth} for path {' -> '.join(current_path)}. Error: {llm_response}")
        all_paths.append({"path": list(current_path), "status": "llm_error", "error_message": llm_response or "No response"})
        return

    parsed_nodes = process_tot_llm_response(llm_response, current_depth, max_depth)

    if current_depth == max_depth: # Reached leaf node (max depth)
        all_paths.append({"path": list(current_path), "status": "completed", "final_node_output": parsed_nodes[0]["conclusion"] if parsed_nodes else "N/A"})
        return

    for node in parsed_nodes:
        new_path = list(current_path)
        new_path.append(node["thought"])
        explore_thought_tree_recursively(problem, api_key, new_path, current_depth + 1, max_depth, all_paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore Tree of Thoughts for a given problem.")
    parser.add_argument("--problem", type=str, required=True, help="The problem statement to explore.")
    parser.add_argument("--max_depth", type=int, default=2, help="Maximum depth of the thought tree.")
    parser.add_argument("--api_key", type=str, default=None, help="API key for the LLM service.")
    args = parser.parse_args()

    api_key = args.api_key or get_deepseek_api_key()
    if not api_key:
        print("Error: API key is required.")
        sys.exit(1)

    all_paths = []
    explore_thought_tree_recursively(args.problem, api_key, [], 0, args.max_depth, all_paths)

    output_dir = os.path.join("..", "..", "DataOutput")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "thai_dataset_tree_of_thoughts.csv")

    rows = []
    for path in all_paths:
        rows.append([str(uuid.uuid4()), json.dumps(path), "tree_of_thoughts"])

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text', 'label'])
        writer.writerows(rows)

    print(f"Created {output_file}")