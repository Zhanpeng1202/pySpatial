#!/usr/bin/env python3
"""
Test pySpatial Agent code generation on MindCube_tinybench.jsonl entries.

This script processes each JSONL entry as a scene and uses the Agent class
to generate code for spatial reasoning problems. The results are saved to
the current directory.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add parent directory to Python path to import pySpatial_Interface
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pySpatial_Interface import Agent, Scene




def make_json_serializable(obj):
    """
    Convert objects to JSON-serializable format.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, 'save'):
        # PIL Image or similar - should have been handled by save_visual_clue
        return f"<Image object: {type(obj).__name__}>"
    else:
        # Convert other objects to string
        return str(obj)


def evaluate_answer_correctness(generated_answer: str, expected_answer: str) -> bool:
    print(f"Generated answer: {generated_answer}")
    print(f"Expected answer: {expected_answer}")

    return generated_answer == expected_answer


def process_scene_with_agent(entry: Dict[str, Any], agent: Agent) -> Dict[str, Any]:
    """
    Process a single JSONL entry through the complete pipeline:
    1. Generate code
    2. Execute code to get visual clue
    3. Generate answer using visual clue
    4. Evaluate correctness

    Args:
        entry: JSONL entry containing scene information
        agent: pySpatial Agent instance

    Returns:
        Dictionary containing the complete pipeline results
    """
    pipeline_start_time = time.time()

    scene_id = entry['id']
    question = entry.get('question', '')
    images = entry.get('images', [])
    expected_answer = entry.get('gt_answer', '')  # Get expected answer if available

    scene = Scene(images, question, scene_id=scene_id)

    # Step 1: Generate code using the agent
    code_gen_start = time.time()
    generated_response = agent.generate_code(scene)
    code_gen_time = time.time() - code_gen_start

    # Parse the response to extract code patterns
    parse_start = time.time()
    parsed_code = agent.parse_LLM_response(scene, generated_response)
    parse_time = time.time() - parse_start
    parse_success = parsed_code is not None and parsed_code.strip() != ""

    visual_clue = None
    generated_answer = None
    answer_correct = False
    execution_success = False
    answer_generation_success = False
    execution_time = 0
    answer_gen_time = 0
    evaluation_time = 0

    # Step 2: Execute code to get visual clue (if parsing was successful)
    if parse_success:
        execution_start = time.time()
        visual_clue = agent.execute(scene)
        execution_time = time.time() - execution_start
        execution_success = visual_clue != "there is an error during code generation, no visual clue provided"

        # Step 3: Generate answer using visual clue
        if execution_success:
            answer_gen_start = time.time()
            answer_response = agent.answer(scene, visual_clue)
            answer_gen_time = time.time() - answer_gen_start
            answer_generation_success = answer_response is not None

            if answer_generation_success:
                generated_answer = answer_response.answer
                generated_reasoning = answer_response.reasoning

                # Step 4: Evaluate correctness
                if expected_answer and generated_answer:
                    eval_start = time.time()
                    answer_correct = evaluate_answer_correctness(generated_answer, expected_answer)
                    evaluation_time = time.time() - eval_start

    total_pipeline_time = time.time() - pipeline_start_time

    result = {
        "scene_id": scene_id,
        "question": question,
        "expected_answer": expected_answer,
        "generated_response": generated_response,
        "parsed_code": parsed_code,
        "parse_success": parse_success,
        "execution_success": execution_success,
        "execution_error": visual_clue if not execution_success and isinstance(visual_clue, str) else None,
        "visual_clue_type": type(visual_clue).__name__ if visual_clue and execution_success else None,
        "answer_generation_success": answer_generation_success,
        "generated_answer": generated_answer,
        "generated_reasoning": generated_reasoning,
        "answer_correct": answer_correct,
        "status": "success",
        "timing": {
            "total_pipeline_time": round(total_pipeline_time, 3),
            "code_generation_time": round(code_gen_time, 3),
            "parse_time": round(parse_time, 3),
            "execution_time": round(execution_time, 3),
            "answer_generation_time": round(answer_gen_time, 3),
            "evaluation_time": round(evaluation_time, 3)
        }
    }


    return result


def main():
    parser = argparse.ArgumentParser(description="Test pySpatial Agent code generation on MindCube scenes")
    parser.add_argument("--jsonl_path", type=str, 
                       default="/data/Datasets/MindCube/data/raw/MindCube_tinybench.jsonl",
                       help="Path to JSONL file containing scene information")
    parser.add_argument("--output_file", type=str,
                       default="agent_code_generation_results_v3.json",
                       help="Output file path for results")
    parser.add_argument("--max_entries", type=int, default=None,
                       help="Maximum number of entries to process (for testing)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API key (if not provided, uses OPENAI_API_KEY env var)")
    parser.add_argument("--translation_only", action="store_true", default=False,
                       help="Only process translation questions (default: process all questions)")
    parser.add_argument("--debug_scene", type=int, default=10,
                       help="Debug mode: skip to specific scene number (e.g., --debug_scene 3)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl_path):
        raise ValueError(f"JSONL file not found: {args.jsonl_path}")
    
    # Initialize Agent
    agent = Agent(api_key=args.api_key)
    print("✓ Agent initialized successfully")
    
    print(f"Processing JSONL file: {args.jsonl_path}")
    print(f"Output file: {args.output_file}")
    print(f"Max entries: {args.max_entries or 'unlimited'}")
    if args.debug_scene:
        print(f"Debug mode: Skipping to scene {args.debug_scene}")
    print("="*60)
    
    # First, read all entries and apply filters to determine which ones to process
    all_entries = []
    with open(args.jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            entry = json.loads(line.strip())
            entry['line_num'] = line_num  # Keep track of original line number

            # Debug mode: skip to specific scene
            if args.debug_scene:
                if line_num < args.debug_scene:
                    continue

            # Check if this is a translation question (if filtering is enabled)
            if args.translation_only:
                question = entry.get('question', '')
                is_translation_question = question.startswith("Based on these two views showing the same scene: in which direction did I move from the first view to the second view")

                # Only process translation questions when filtering is enabled
                if not is_translation_question:
                    continue

            all_entries.append(entry)

    # Calculate which entries to process with even spacing
    total_entries = len(all_entries)
    if args.max_entries and args.max_entries < total_entries:
        # Calculate step size to get even spacing
        step_size = total_entries // args.max_entries
        if step_size == 0:
            step_size = 1

        # Select entries at even intervals
        selected_entries = all_entries[::step_size][:args.max_entries]
        print(f"Selected {len(selected_entries)} entries out of {total_entries} with step size {step_size}")
    else:
        selected_entries = all_entries
        print(f"Processing all {len(selected_entries)} entries")

    # Process selected entries
    processed_count = 0
    successful_count = 0
    failed_count = 0
    parse_success_count = 0
    execution_success_count = 0
    answer_success_count = 0
    correct_answer_count = 0
    evaluable_answer_count = 0  # Answers where expected answer is available
    results = []

    for entry in selected_entries:
        processed_count += 1

        print(f"\n--- Processing entry {processed_count}/{len(selected_entries)}: {entry.get('id', 'unknown')} (original line {entry['line_num']}) ---")

        # Process scene with agent
        result = process_scene_with_agent(entry, agent)
        results.append(result)

        # Update pipeline step counters
        if result.get("parse_success", False):
            parse_success_count += 1
        if result.get("execution_success", False):
            execution_success_count += 1
        if result.get("answer_generation_success", False):
            answer_success_count += 1
        if result.get("expected_answer") and result.get("generated_answer"):
            evaluable_answer_count += 1
            if result.get("answer_correct", False):
                correct_answer_count += 1

        successful_count += 1
                    
    
    # Save results to current directory
    output_path = Path.cwd() / args.output_file
    
    # Calculate detailed metrics
    parse_rate = parse_success_count / processed_count * 100 if processed_count > 0 else 0
    execution_rate = execution_success_count / processed_count * 100 if processed_count > 0 else 0
    answer_gen_rate = answer_success_count / processed_count * 100 if processed_count > 0 else 0
    correctness_rate = correct_answer_count / evaluable_answer_count * 100 if evaluable_answer_count > 0 else 0
    overall_success_rate = successful_count / processed_count * 100 if processed_count > 0 else 0

    # Calculate timing statistics
    timing_stats = {}
    if results:
        timing_keys = ["total_pipeline_time", "code_generation_time", "parse_time",
                      "execution_time", "answer_generation_time", "evaluation_time"]

        for key in timing_keys:
            times = [r["timing"][key] for r in results if "timing" in r and key in r["timing"]]
            if times:
                timing_stats[key] = {
                    "mean": round(sum(times) / len(times), 3),
                    "min": round(min(times), 3),
                    "max": round(max(times), 3),
                    "total": round(sum(times), 3)
                }

    metrics = {
        "total_processed": processed_count,
        "overall_success": successful_count,
        "overall_failed": failed_count,
        "parse_success_count": parse_success_count,
        "execution_success_count": execution_success_count,
        "answer_generation_success_count": answer_success_count,
        "evaluable_answer_count": evaluable_answer_count,
        "correct_answer_count": correct_answer_count,
        "rates": {
            "overall_success_rate": round(overall_success_rate, 2),
            "parse_success_rate": round(parse_rate, 2),
            "execution_success_rate": round(execution_rate, 2),
            "answer_generation_rate": round(answer_gen_rate, 2),
            "answer_correctness_rate": round(correctness_rate, 2)
        },
        "timing_statistics": timing_stats
    }
    
    summary = {
        "metrics": metrics,
        "processing_timestamp": datetime.now().isoformat(),
        "jsonl_source": args.jsonl_path,
        "results": results
    }
    
    # Make sure all data is JSON serializable
    serializable_summary = make_json_serializable(summary)
    with open(output_path, 'w') as f:
        json.dump(serializable_summary, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")
    
    print(f"\n=== Complete Pipeline Results ===")
    print(f"Total entries processed: {processed_count}")
    print(f"Overall pipeline success: {successful_count} ({overall_success_rate:.1f}%)")
    print(f"Failed entries: {failed_count}")
    print(f"\n=== Step-by-Step Breakdown ===")
    print(f"Code parsing success: {parse_success_count}/{processed_count} ({parse_rate:.1f}%)")
    print(f"Code execution success: {execution_success_count}/{processed_count} ({execution_rate:.1f}%)")
    print(f"Answer generation success: {answer_success_count}/{processed_count} ({answer_gen_rate:.1f}%)")
    print(f"Answer correctness: {correct_answer_count}/{evaluable_answer_count} ({correctness_rate:.1f}%)" if evaluable_answer_count > 0 else "Answer correctness: N/A (no expected answers)")

    # Print timing statistics
    if timing_stats:
        print(f"\n=== Timing Statistics (seconds) ===")
        for key, stats in timing_stats.items():
            key_display = key.replace('_', ' ').title()
            print(f"{key_display}: avg={stats['mean']}s, min={stats['min']}s, max={stats['max']}s, total={stats['total']}s")


if __name__ == "__main__":
    main()