from gpt_client_bank import *
from datasets import load_dataset
from typing import Dict, List, Tuple
import json
from datetime import datetime
import os
import time
import argparse

# Create results directory if not exists
os.makedirs("./results", exist_ok=True)

# Define all difficulty splits
DIFFICULTY_SPLITS = [
    ("extremely_hard_0.0_0.1", "Extremely Hard (0.0-0.1)"),
    ("very_hard_0.1_0.2", "Very Hard (0.1-0.2)"),
    ("hard_0.2_0.3", "Hard (0.2-0.3)"),
    ("moderately_hard_0.3_0.4", "Moderately Hard (0.3-0.4)"),
    ("intermediate_0.4_0.5", "Intermediate (0.4-0.5)"),
    ("medium_0.5_0.6", "Medium (0.5-0.6)"),
    ("moderately_easy_0.6_0.7", "Moderately Easy (0.6-0.7)"),
    ("easy_0.7_0.8", "Easy (0.7-0.8)"),
    ("very_easy_0.8_0.9", "Very Easy (0.8-0.9)"),
    ("extremely_easy_0.9_1.0", "Extremely Easy (0.9-1.0)")
]

def load_all_splits():
    """Load all difficulty splits of the dataset"""
    return [(split, load_dataset("wzzzq/MMLU-PRO-Leveled-TinyBench", split=split))
            for split, name in DIFFICULTY_SPLITS]

def format_question(entry):
    question = entry["question"]
    options = entry["options"]
    category = entry["category"]
    
    prompt = (
        f"IMPORTANT: This is a {category} multiple-choice question with options labeled A-J. "
        "You MUST:\n"
        "1. Analyze the question carefully\n"
        "2. Choose ONLY ONE correct option\n"
        "3. Output JUST THE SINGLE LETTER (A-J)\n\n"
        "STRICT RULES:\n"
        "- DO NOT EXPLAIN YOUR REASONING\n"
        "- DO NOT WRITE ANYTHING ELSE\n"
        "- DO NOT USE PUNCTUATION AFTER THE LETTER\n"
        "- YOUR ANSWER WILL BE REJECTED IF IT CONTAINS EXTRA TEXT\n\n"
        f"Question: {question}\n"
        "Options:\n"
    )
    
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        prompt += f"{choice_map[i]}. {opt}\n"
    
    prompt += (
        "\nREITERATION: After careful consideration, the correct answer is: "
        "[ONLY YOUR FINAL LETTER CHOICE HERE].\n"
        "Answer: "
    )
    return prompt

def evaluate_model(client, dataset_split,num=10) -> Tuple[float, List[Dict]]:
    """Evaluate a single model on a dataset split"""
    correct = 0
    total = 0
    results = []

    for problem in dataset_split:
        if total>=num:
            break
        question = format_question(problem)
        response = client.send_text(question)
        
        predicted_answer = response.strip()[0].upper() if response else ""
        correct_answer = problem['answer']
        
        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'question_id': problem['question_id'],
            'difficulty': problem['difficulty'],
            'category': problem['category'],
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'response': response
        })

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results

def initialize_report_files(timestamp: str) -> Tuple[str, str]:
    """Initialize output files and return their paths"""
    json_path = f"./results/evaluation_full_{timestamp}.json"
    md_path = f"./results/evaluation_report_{timestamp}.md"
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Model Performance Across Difficulty Levels\n\n")
        headers = ["Model"] + [split[1] for split in DIFFICULTY_SPLITS] + ["Average"]
        f.write(f"| {' | '.join(headers)} |\n")
        f.write(f"|{'|'.join(['-------'] * len(headers))}|\n")
    
    return json_path, md_path

def process_single_split(client, split: Tuple[str, str], all_splits: List,num, max_retries: int = 3) -> Tuple[float, List[Dict]]:
    """Process a single difficulty split for a model with retry logic"""
    split_key, display_name = split
    dataset = next((ds for s, ds in all_splits if s == split_key), None)
    
    if not dataset:
        print(f"Warning: Missing dataset for {split_key}")
        return 0.0, []
    
    print(f"Processing split: {display_name}")
    
    retries = 0
    while retries < max_retries:
        try:
            accuracy, split_results = evaluate_model(client, dataset,num)
            print(f"Split accuracy: {accuracy*100:.2f}%")
            return accuracy, split_results
        except Exception as e:
            retries += 1
            if retries < max_retries:
                print(f"Error processing split (attempt {retries}/{max_retries}): {str(e)}")
                print(f"Retrying...")
                time.sleep(2)  # Add a small delay before retrying
            else:
                print(f"Failed after {max_retries} attempts for split {display_name}: {str(e)}")
                return 0.0, []


def evaluate_single_model(model_name: str, client, all_splits: List,num) -> Dict:
    """Evaluate a single model across all difficulty splits"""
    print(f"\n{'='*40}\nTesting model: {model_name}\n{'='*40}")
    
    model_results = {'splits': {}, 'details': {}}
    
    for split in DIFFICULTY_SPLITS:
        accuracy, split_results = process_single_split(client, split, all_splits,num)
        model_results['splits'][split[0]] = accuracy
        model_results['details'][split[0]] = split_results
    
    return model_results

def save_results_to_json(results: Dict, json_path: str):
    """Save current results state to JSON file"""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def calculate_average_accuracy(accuracies: List[float]) -> float:
    """Calculate average accuracy from a list of split accuracies"""
    return sum(accuracies)/len(accuracies) if accuracies else 0.0

def generate_markdown_row(model_name: str, model_data: Dict) -> str:
    """Generate a markdown table row for a model's results"""
    accuracies = []
    row = [model_name]
    
    for split_key, _ in DIFFICULTY_SPLITS:
        accuracy = model_data['splits'].get(split_key, 0.0)
        accuracies.append(accuracy)
        row.append(f"{accuracy*100:.2f}%")
    
    avg_accuracy = calculate_average_accuracy(accuracies)
    row.append(f"{avg_accuracy*100:.2f}%")
    
    return f"| {' | '.join(row)} |\n"

if __name__ == "__main__":
    # Initialization
    parser = argparse.ArgumentParser(description='Tiny Benchmark Test')
    parser.add_argument('--num', type=int, default=10, help='')
    args = parser.parse_args()
    num=args.num

    all_splits = load_all_splits()
    client_dict = get_gpt_client_dict()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path, md_path = initialize_report_files(timestamp)
    results = {}

    # Main evaluation pipeline
    for model_name, client in client_dict.items():
        # Evaluate model across all splits
        model_results = evaluate_single_model(model_name, client, all_splits,num)
        results[model_name] = model_results
        
        # Persist results
        save_results_to_json(results, json_path)
        
        # Update markdown report
        md_row = generate_markdown_row(model_name, model_results)
        with open(md_path, 'a', encoding='utf-8') as f:
            f.write(md_row)

    print(f"\nSaved results to:\n- {json_path}\n- {md_path}")
