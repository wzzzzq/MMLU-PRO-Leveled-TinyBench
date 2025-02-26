from gpt_client_bank import *
from datasets import load_dataset
from typing import Dict, List, Tuple
import json
from datetime import datetime

def load_test_data():
    """Load the hardest split of MMLU-PRO dataset"""
    return load_dataset("wzzzq/MMLU-PRO-Leveled-TinyBench", split="extremely_hard_0.0_0.1")

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



def evaluate_model(client, dataset) -> Tuple[float, List[Dict]]:
    """Evaluate a single model on the dataset"""
    correct = 0
    total = 0
    results = []
    
    for problem in dataset:
        question = format_question(problem)
        response = client.send_text(question)
        
        # Extract the answer (first letter in the response)
        predicted_answer = response.strip()[0].upper() if response else ""
        correct_answer = problem['answer']
        
        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct += 1
        total += 1
        
        # Enhanced result logging
        results.append({
            'question_id': problem['question_id'],  # Assuming dataset has 'id' field
            'question': problem['question'],
            'difficulty': problem['difficulty'],  # Assuming dataset has 'difficulty' field
            'category': problem['category'],
            'options': problem['options'],
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'full_response': response,
        })
        
        print(f"Progress: {total} questions processed. Current accuracy: {(correct/total)*100:.2f}%")
    
    accuracy = correct / total
    return accuracy, results

def generate_markdown_table(results: Dict) -> str:
    """Generate markdown table from evaluation results"""
    md = "# Model Evaluation Results\n\n"
    md += "| Model Name | Accuracy |\n"
    md += "|------------|----------|\n"
    
    # Sort models by accuracy descending
    sorted_models = sorted(results.items(), 
                         key=lambda x: x[1]['accuracy'], 
                         reverse=True)
    
    for model_name, data in sorted_models:
        accuracy = data['accuracy']
        md += f"| {model_name} | {accuracy*100:.2f}% |\n"
    
    return md

if __name__ == "__main__":
    """Main function to run the evaluation"""
    dataset = load_test_data()
    client_dict = get_gpt_client_dict()
    
    results = {}
    
    for model_name, client in client_dict.items():
        print(f"\nTesting model: {model_name}")
        try:
            accuracy, model_results = evaluate_model(client, dataset)
            results[model_name] = {
                'accuracy': accuracy,
                'results': model_results
            }
            print(f"{model_name} final accuracy: {accuracy*100:.2f}%")
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
    
    # Save results to files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_filename = f'evaluation_results_{timestamp}.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # Save Markdown
    md_filename = f'evaluation_report_{timestamp}.md'
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(generate_markdown_table(results))
    
    print(f"\nSaved results to {json_filename} and {md_filename}")