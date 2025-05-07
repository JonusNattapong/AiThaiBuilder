import os
import sys
import json
import argparse
import random
from tqdm import tqdm
import re

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from utils.deepseek_utils import generate_with_deepseek, get_deepseek_api_key

# Mock tool functions for the LLM to "use"
# In a real ReAct system, these would interact with actual APIs or environments.
def mock_search_tool(query: str) -> str:
    """Simulates a search engine."""
    if "capital of france" in query.lower():
        return "Paris is the capital of France."
    if "weather in london" in query.lower():
        return "The weather in London is currently 15°C and cloudy."
    return f"Search results for '{query}': No specific information found in mock tool."

def mock_calculator_tool(expression: str) -> str:
    """Simulates a calculator."""
    try:
        # VERY basic and unsafe eval. Use a proper math parser in production.
        if not re.match(r"^[0-9+\-*/().\s]+$", expression):
            return "Error: Invalid characters in expression."
        result = eval(expression)
        return f"Result of '{expression}' is {result}."
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"

AVAILABLE_TOOLS = {
    "Search": mock_search_tool,
    "Calculator": mock_calculator_tool
}

def generate_react_prompt(task_description: str):
    """
    Generates prompts for ReAct-style reasoning.
    """
    system_prompt = (
        "You are an AI assistant that solves tasks by reasoning and taking actions. "
        "You have access to the following tools: Search[query], Calculator[expression].\n"
        "Follow the ReAct format: Thought, Action, Observation, Thought, Action, Observation, ... Final Answer."
    )
    user_prompt = (
        f"Task: {task_description}\n\n"
        "Please solve this task using the ReAct framework. Show your thought process, the actions you take (e.g., Search[your query] or Calculator[your expression]), "
        "the observations you make from those actions, and then your final answer.\n"
        "Start with a 'Thought:' and end with 'Final Answer: <your answer>'."
    )
    return system_prompt, user_prompt

def process_react_response(task: str, llm_response: str):
    """
    Processes the LLM's response for ReAct data.
    This is a simplified parser. Robust ReAct parsing can be complex.
    """
    # For now, we'll store the raw trace.
    # A more advanced parser would extract each Thought/Action/Observation step.
    # It might also simulate tool calls if the LLM generates them.
    
    # Example of how one might try to parse steps (very basic):
    steps = []
    current_response_lines = llm_response.split('\n')
    for line in current_response_lines:
        if line.startswith("Thought:"):
            steps.append({"type": "thought", "content": line.replace("Thought:", "").strip()})
        elif line.startswith("Action:"):
            action_content = line.replace("Action:", "").strip()
            steps.append({"type": "action", "content": action_content})
            # Potentially simulate the action here if it's a known tool
            tool_match = re.match(r"(\w+)\[(.*)\]", action_content)
            if tool_match:
                tool_name, tool_input = tool_match.groups()
                if tool_name in AVAILABLE_TOOLS:
                    observation = AVAILABLE_TOOLS[tool_name](tool_input)
                    steps.append({"type": "observation", "content": observation})
                else:
                    steps.append({"type": "observation", "content": f"Error: Tool '{tool_name}' not available."})
            else: # if action is not a tool call, it might be an error or different format
                steps.append({"type": "observation", "content": "Observation: Action was not a recognized tool call."})


        elif line.startswith("Observation:"): # If LLM generates its own observation
            steps.append({"type": "observation", "content": line.replace("Observation:", "").strip()})
        elif line.startswith("Final Answer:"):
            steps.append({"type": "final_answer", "content": line.replace("Final Answer:", "").strip()})
            break # Stop after final answer

    return {
        "task": task,
        "react_trace_simplified": steps, # Store the raw trace for now
        "raw_response": llm_response
    }

def generate_dataset(output_file: str, num_samples: int, api_key: str):
    """
    Generates a dataset for ReAct (Reasoning and Acting).
    """
    # TODO: Define tasks that require tool use or multi-step interaction.
    example_tasks = [
        "What is the capital of France and what is 5 + 7?",
        "Search for the current weather in London and then calculate 100 / 5.",
        "Who was the first president of the USA and what is 25 * 4?"
    ]
    if not example_tasks:
        print("Error: No example tasks for ReAct.")
        return

    generated_samples = []
    for i in tqdm(range(num_samples), desc="Generating ReAct Samples"):
        task = random.choice(example_tasks)
        
        system_prompt, user_prompt = generate_react_prompt(task)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        # ReAct traces can be long
        llm_response = generate_with_deepseek(messages, api_key, max_tokens=1500) 

        if llm_response and not llm_response.startswith("Error:"):
            processed_sample = process_react_response(task, llm_response)
            generated_samples.append(processed_sample)
        else:
            print(f"Warning: Failed to generate valid ReAct response for task: {task}. Error: {llm_response}")
            generated_samples.append({
                "task": task,
                "react_trace_simplified": "GENERATION_ERROR",
                "raw_response": llm_response or "No response"
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in generated_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nSuccessfully generated {len(generated_samples)} ReAct samples and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate ReAct (Reason+Act) dataset.")
    parser.add_argument("--output_file", type=str, default="data/cooked/react_dataset.jsonl", help="Path to save the dataset.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate.")
    parser.add_argument("--api_key", type=str, default=None, help="DeepSeek API Key.")
    
    args = parser.parse_args()
    api_key = get_deepseek_api_key(args.api_key)
    if not api_key:
        print("Error: DeepSeek API Key not found.")
        return
    generate_dataset(args.output_file, args.num_samples, api_key)

if __name__ == "__main__":
    main()

