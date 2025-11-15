#!/usr/bin/env python3
"""
Analyze omni3d results and categorize string questions into true-or-false vs multi-choice
"""

import json
import sys
from pathlib import Path

def categorize_question_type(result):
    """
    Categorize questions into refined types:
    - float: numerical ratios/measurements
    - int: counting questions
    - true_or_false: yes/no questions
    - multi_choice: multiple choice questions
    """
    answer_type = result['answer_type']

    if answer_type in ['float', 'int']:
        return answer_type
    elif answer_type == 'str':
        ground_truth = str(result['ground_truth']).strip().lower()
        if ground_truth in ['yes', 'no']:
            return 'true_or_false'
        else:
            return 'multi_choice'
    else:
        return 'unknown'

def analyze_results(results_path):
    """Analyze results and calculate refined accuracy metrics"""

    with open(results_path, 'r') as f:
        data = json.load(f)

    results = data['results']

    # Initialize counters for refined categories
    correct_by_type = {"float": 0, "int": 0, "true_or_false": 0, "multi_choice": 0}
    total_by_type = {"float": 0, "int": 0, "true_or_false": 0, "multi_choice": 0}
    error_count = 0

    # Categorize and count results
    categorized_results = []
    for result in results:
        if result.get('error'):
            error_count += 1
            continue

        refined_type = categorize_question_type(result)

        # Add refined type to result
        result_with_type = result.copy()
        result_with_type['refined_answer_type'] = refined_type
        categorized_results.append(result_with_type)

        # Update counters
        if refined_type in total_by_type:
            total_by_type[refined_type] += 1
            if result['correct']:
                correct_by_type[refined_type] += 1

    # Calculate refined metrics
    refined_metrics = {}
    for answer_type in ["float", "int", "true_or_false", "multi_choice"]:
        if total_by_type[answer_type] > 0:
            refined_metrics[f"{answer_type}_accuracy"] = correct_by_type[answer_type] / total_by_type[answer_type]
            refined_metrics[f"{answer_type}_count"] = total_by_type[answer_type]
            refined_metrics[f"{answer_type}_correct"] = correct_by_type[answer_type]
        else:
            refined_metrics[f"{answer_type}_accuracy"] = 0.0
            refined_metrics[f"{answer_type}_count"] = 0
            refined_metrics[f"{answer_type}_correct"] = 0

    total_valid = sum(total_by_type.values())
    if total_valid > 0:
        refined_metrics["overall_accuracy"] = sum(correct_by_type.values()) / total_valid
    else:
        refined_metrics["overall_accuracy"] = 0.0

    refined_metrics["total_questions"] = total_valid
    refined_metrics["error_count"] = error_count

    # Keep original metrics for comparison
    original_metrics = data['metrics']

    return {
        'original_metrics': original_metrics,
        'refined_metrics': refined_metrics,
        'categorized_results': categorized_results,
        'breakdown_analysis': {
            'total_str_questions': original_metrics.get('str_count', 0),
            'true_or_false_questions': refined_metrics['true_or_false_count'],
            'multi_choice_questions': refined_metrics['multi_choice_count']
        }
    }

def print_analysis(analysis):
    """Print detailed analysis"""

    print("="*60)
    print("REFINED QUESTION TYPE ANALYSIS")
    print("="*60)

    # Original vs Refined breakdown
    print("\nOriginal vs Refined Categorization:")
    print(f"Original 'str' questions: {analysis['breakdown_analysis']['total_str_questions']}")
    print(f"  ├─ True-or-False: {analysis['breakdown_analysis']['true_or_false_questions']}")
    print(f"  └─ Multi-Choice: {analysis['breakdown_analysis']['multi_choice_questions']}")

    # Accuracy by refined type
    print(f"\nAccuracy by Refined Question Type:")
    refined = analysis['refined_metrics']

    for question_type in ["float", "int", "true_or_false", "multi_choice"]:
        acc = refined[f"{question_type}_accuracy"]
        correct = refined[f"{question_type}_correct"]
        total = refined[f"{question_type}_count"]
        print(f"  {question_type.replace('_', '-').upper()}: {acc:.3f} ({correct}/{total})")

    print(f"\nOverall Accuracy: {refined['overall_accuracy']:.3f} ({sum([refined[f'{t}_correct'] for t in ['float', 'int', 'true_or_false', 'multi_choice']])}/{refined['total_questions']})")

    # Comparison with original
    print(f"\nComparison with Original Metrics:")
    original = analysis['original_metrics']
    print(f"Original overall accuracy: {original['overall_accuracy']:.3f}")
    print(f"Refined overall accuracy:  {refined['overall_accuracy']:.3f}")

    print(f"\nOriginal string accuracy: {original.get('str_accuracy', 0):.3f} ({original.get('str_count', 0)} questions)")
    print(f"Refined true-or-false:    {refined['true_or_false_accuracy']:.3f} ({refined['true_or_false_count']} questions)")
    print(f"Refined multi-choice:     {refined['multi_choice_accuracy']:.3f} ({refined['multi_choice_count']} questions)")

def main():
    results_path = "omni3d_results.json"

    if len(sys.argv) > 1:
        results_path = sys.argv[1]

    if not Path(results_path).exists():
        print(f"Error: Results file not found: {results_path}")
        return

    analysis = analyze_results(results_path)
    print_analysis(analysis)

    # Save refined results
    output_path = results_path.replace('.json', '_refined.json')

    refined_data = {
        'original_metrics': analysis['original_metrics'],
        'refined_metrics': analysis['refined_metrics'],
        'breakdown_analysis': analysis['breakdown_analysis'],
        'results': analysis['categorized_results']
    }

    with open(output_path, 'w') as f:
        json.dump(refined_data, f, indent=2)

    print(f"\nRefined analysis saved to: {output_path}")

if __name__ == "__main__":
    main()