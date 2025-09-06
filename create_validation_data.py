#!/usr/bin/env python3
"""
Script to create validation.jsonl from MATH test dataset.
This converts the individual JSON files in the test directory to a single JSONL file
with the format expected by eval_zero_shot.py.
"""

import json
import os
import re
from pathlib import Path

def get_solution_as_answer(solution):
    """
    Use the entire solution as the answer without any processing
    """
    return solution

def create_validation_jsonl():
    """
    Create separate train.jsonl and validation.jsonl files from MATH train and test directories
    """
    math_train_dir = Path("data/MATH/train")
    math_test_dir = Path("data/MATH/test")
    
    # Create output directory if it doesn't exist
    os.makedirs("data/MATH", exist_ok=True)
    
    train_problems = []
    val_problems = []
    
    # Process train directory
    if math_train_dir.exists():
        print(f"Processing train data from {math_train_dir}...")
        for subdir in math_train_dir.iterdir():
            if subdir.is_dir():
                print(f"  Processing {subdir.name}...")
                for json_file in subdir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Use the entire solution as the answer
                        answer = get_solution_as_answer(data['solution'])
                        
                        if answer:  # Only include if we found an answer
                            problem_data = {
                                'problem': data['problem'],
                                'answer': answer,
                                'level': data.get('level', ''),
                                'type': data.get('type', ''),
                                'solution': data['solution']
                            }
                            train_problems.append(problem_data)
                        else:
                            print(f"    Skipping {json_file} - no answer found")
                            
                    except Exception as e:
                        print(f"    Error processing {json_file}: {e}")
    else:
        print(f"Warning: {math_train_dir} does not exist")
    
    # Process test directory (use as validation)
    if math_test_dir.exists():
        print(f"Processing validation data from {math_test_dir}...")
        for subdir in math_test_dir.iterdir():
            if subdir.is_dir():
                print(f"  Processing {subdir.name}...")
                for json_file in subdir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Use the entire solution as the answer
                        answer = get_solution_as_answer(data['solution'])
                        
                        if answer:  # Only include if we found an answer
                            problem_data = {
                                'problem': data['problem'],
                                'answer': answer,
                                'level': data.get('level', ''),
                                'type': data.get('type', ''),
                                'solution': data['solution']
                            }
                            val_problems.append(problem_data)
                        else:
                            print(f"    Skipping {json_file} - no answer found")
                            
                    except Exception as e:
                        print(f"    Error processing {json_file}: {e}")
    else:
        print(f"Warning: {math_test_dir} does not exist")
    
    print(f"Found {len(train_problems)} train problems with valid answers")
    print(f"Found {len(val_problems)} validation problems with valid answers")
    
    # Write train data to JSONL file
    train_output_file = "data/MATH/train.jsonl"
    with open(train_output_file, 'w', encoding='utf-8') as f:
        for problem in train_problems:
            json_line = {
                'problem': problem['problem'],
                'answer': problem['answer']
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
    
    # Write validation data to JSONL file
    val_output_file = "data/MATH/validation.jsonl"
    with open(val_output_file, 'w', encoding='utf-8') as f:
        for problem in val_problems:
            json_line = {
                'problem': problem['problem'],
                'answer': problem['answer']
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
    
    print(f"Created {train_output_file} with {len(train_problems)} problems")
    print(f"Created {val_output_file} with {len(val_problems)} problems")
    
    # Show a sample of the created files
    print("\nSample entries from validation.jsonl:")
    with open(val_output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 3:  # Show first 3 entries
                data = json.loads(line)
                print(f"Entry {i+1}:")
                print(f"  Problem: {data['problem'][:100]}...")
                print(f"  Answer: {data['answer']}")
                print()
    
    print("Sample entries from train.jsonl:")
    with open(train_output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 3:  # Show first 3 entries
                data = json.loads(line)
                print(f"Entry {i+1}:")
                print(f"  Problem: {data['problem'][:100]}...")
                print(f"  Answer: {data['answer']}")
                print()

if __name__ == "__main__":
    create_validation_jsonl()
