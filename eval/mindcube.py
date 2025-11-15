
#!/usr/bin/env python3
"""
Evaluate pySpatial Agent on MindCube dataset and calculate statistics for three types:
- among: from image paths like "other_all_image/among/shoe_216/front_007.jpg"
- around: from image paths like "other_all_image/around/26b1a4b226e2e3509100a595ebc5d17dafd361abfdf06fcf20e36f905e138faa/2_frame_00166.png"
- rotation: from image paths containing "rotation"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import threading
import backoff

# Add parent directory to Python path to import pySpatial_Interface
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pySpatial_Interface import Agent, Scene

# Rate limiting globals
last_request_time = 0
min_request_interval = 0.1  # Minimum time between requests (100ms)
request_lock = threading.Lock()


def rate_limit():
    """Apply rate limiting between API requests"""
    global last_request_time
    
    with request_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < min_request_interval:
            time.sleep(min_request_interval - time_since_last)
        last_request_time = time.time()


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    factor=2,
    jitter=backoff.full_jitter,
    on_backoff=lambda details: print(f"Retrying {details['target'].__name__} (attempt {details['tries']}/{details['max_tries']})...")
)
def call_agent_with_retry(agent, method_name, *args, **kwargs):
    """Call agent method with rate limiting and retry logic"""
    rate_limit()
    
    try:
        method = getattr(agent, method_name)
        result = method(*args, **kwargs)
        return result
    except Exception as e:
        error_msg = str(e)
        
        # Handle rate limit specifically
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            print(f"Rate limit hit for {method_name}, waiting before retry...")
            time.sleep(60) 
        elif "tokens per min" in error_msg.lower():
            print(f"Token rate limit hit for {method_name}, waiting before retry...")
            time.sleep(60)  
            raise  
        else:
            # For other errors, re-raise without modification
            raise


def extract_type_from_images(images: List[str]) -> str:
    """
    Extract the type (among, around, rotation) from the image paths.
    
    Args:
        images: List of image paths
        
    Returns:
        The type string or 'unknown' if cannot be determined
    """
    for image_path in images:
        if 'among' in image_path:
            return 'among'
        elif 'around' in image_path:
            return 'around'
        elif 'rotation' in image_path:
            return 'rotation'
    
    return 'unknown'


def evaluate_answer_correctness(generated_answer: str, expected_answer: str) -> bool:
    """Check if generated answer matches expected answer."""
    return generated_answer == expected_answer


def process_scene_with_agent_wrapper(args_tuple) -> Dict[str, Any]:
    """
    Wrapper function for multiprocessing that creates its own agent instance.
    
    Args:
        args_tuple: Tuple of (entry, api_key)
        
    Returns:
        Dictionary containing the complete pipeline results including type information
    """
    entry, api_key = args_tuple
    
    # Create agent instance for this process
    agent = Agent(api_key=api_key)
    
    return process_scene_with_agent(entry, agent)


def process_scene_with_agent(entry: Dict[str, Any], agent: Agent) -> Dict[str, Any]:
    """
    Process a single JSONL entry through the complete pipeline and extract type information.
    
    Args:
        entry: JSONL entry containing scene information
        agent: pySpatial Agent instance
        
    Returns:
        Dictionary containing the complete pipeline results including type information
    """
    scene_id = entry['id']
    question = entry.get('question', '')
    images = entry.get('images', [])
    expected_answer = entry.get('gt_answer', '')
    
    # Extract type from image paths
    scene_type = extract_type_from_images(images)
    
    scene = Scene(images, question, scene_id=scene_id)
    
    try:
        # Step 1: Generate code using the agent (with retry)
        generated_response = call_agent_with_retry(agent, 'generate_code', scene)
        
        # Parse the response to extract code patterns
        parsed_code = agent.parse_LLM_response(scene, generated_response)
        parse_success = parsed_code is not None and parsed_code.strip() != ""
        
        visual_clue = None
        generated_answer = None
        answer_correct = False
        execution_success = False
        answer_generation_success = False
        
        # Step 2: Execute code to get visual clue (if parsing was successful)
        if parse_success:
            visual_clue = agent.execute(scene)
            execution_success = visual_clue != "there is an error during code generation, no visual clue provided"
            
            # Step 3: Generate answer using visual clue (with retry)
            if execution_success:
                answer_response = call_agent_with_retry(agent, 'answer', scene, visual_clue)
                answer_generation_success = answer_response is not None
                
                if answer_generation_success:
                    generated_answer = answer_response.answer
                    
                    # Step 4: Evaluate correctness
                    if expected_answer and generated_answer:
                        answer_correct = evaluate_answer_correctness(generated_answer, expected_answer)
        
        result = {
            "scene_id": scene_id,
            "scene_type": scene_type,
            "question": question,
            "images": images,
            "expected_answer": expected_answer,
            "parse_success": parse_success,
            "execution_success": execution_success,
            "answer_generation_success": answer_generation_success,
            "generated_answer": generated_answer,
            "answer_correct": answer_correct,
        }
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing {scene_id}: {error_msg}")
        
        # Return error result
        return {
            "scene_id": scene_id,
            "scene_type": scene_type,
            "question": question,
            "images": images,
            "expected_answer": expected_answer,
            "parse_success": False,
            "execution_success": False,
            "answer_generation_success": False,
            "generated_answer": None,
            "answer_correct": False,
            "error": error_msg,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate pySpatial Agent on MindCube dataset with type statistics")
    parser.add_argument("--jsonl_path", type=str, 
                       default="/data/Datasets/MindCube/data/raw/MindCube_tinybench.jsonl",
                       help="Path to JSONL file containing scene information")
    parser.add_argument("--output_file", type=str,
                       default="mindcube_eval_results.json",
                       help="Output file path for results")
    parser.add_argument("--max_entries", type=int, default=None,
                       help="Maximum number of entries to process")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"),
                       help="OpenAI API key (if not provided, uses OPENAI_API_KEY env var)")
    parser.add_argument("--num_processes", type=int, default=16,
                       help="Number of processes to use (default: number of CPU cores)")
    parser.add_argument("--disable_multiprocessing", action="store_true", 
                       help="Disable multiprocessing and run sequentially")
    parser.add_argument("--request_interval", type=float, default=0.1,
                       help="Minimum time between API requests in seconds (default: 0.1)")
    
    args = parser.parse_args()
    
    # Update global rate limiting interval
    global min_request_interval
    min_request_interval = args.request_interval
    
    if not os.path.exists(args.jsonl_path):
        raise ValueError(f"JSONL file not found: {args.jsonl_path}")
    
    # Determine number of processes
    if args.disable_multiprocessing:
        num_processes = 1
    else:
        num_processes = args.num_processes or cpu_count()
    
    print(f"Processing JSONL file: {args.jsonl_path}")
    print(f"Output file: {args.output_file}")
    print(f"Max entries: {args.max_entries or 'all'}")
    print(f"Number of processes: {num_processes}")
    print(f"Request interval: {min_request_interval}s")
    print("="*60)
    
    # Load all entries first
    entries = []
    with open(args.jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if args.max_entries and len(entries) >= args.max_entries:
                print(f"Reached maximum entries limit: {args.max_entries}")
                break
            
            entry = json.loads(line.strip())
            entries.append(entry)
    
    print(f"Loaded {len(entries)} entries for processing")
    
    # Process entries
    start_time = time.time()
    
    if num_processes == 1 or args.disable_multiprocessing:
        # Sequential processing
        print("Running sequentially...")
        agent = Agent(api_key=args.api_key)
        results = []
        for i, entry in enumerate(entries, 1):
            print(f"Processing entry {i}/{len(entries)}: {entry.get('id', 'unknown')}")
            result = process_scene_with_agent(entry, agent)
            results.append(result)
    else:
        # Multiprocessing
        print(f"Running with {num_processes} processes...")
        
        # Prepare arguments for multiprocessing
        args_list = [(entry, args.api_key) for entry in entries]
        
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_scene_with_agent_wrapper, args_list)
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\n✓ Processing completed in {processing_time:.2f} seconds")
    print(f"Average time per entry: {processing_time/len(entries):.2f} seconds")
    
    # Calculate statistics
    type_stats = defaultdict(lambda: {
        'total': 0,
        'parse_success': 0,
        'execution_success': 0,
        'answer_generation_success': 0,
        'correct_answers': 0,
        'evaluable_answers': 0,
        'errors': 0
    })
    
    overall_stats = {
        'total_processed': 0,
        'parse_success': 0,
        'execution_success': 0,
        'answer_generation_success': 0,
        'correct_answers': 0,
        'evaluable_answers': 0,
        'errors': 0
    }
    
    for result in results:
        scene_type = result['scene_type']
        
        # Update statistics
        type_stats[scene_type]['total'] += 1
        overall_stats['total_processed'] += 1
        
        if result.get('error'):
            type_stats[scene_type]['errors'] += 1
            overall_stats['errors'] += 1
        
        if result['parse_success']:
            type_stats[scene_type]['parse_success'] += 1
            overall_stats['parse_success'] += 1
        
        if result['execution_success']:
            type_stats[scene_type]['execution_success'] += 1
            overall_stats['execution_success'] += 1
        
        if result['answer_generation_success']:
            type_stats[scene_type]['answer_generation_success'] += 1
            overall_stats['answer_generation_success'] += 1
        
        if result['expected_answer'] and result['generated_answer']:
            type_stats[scene_type]['evaluable_answers'] += 1
            overall_stats['evaluable_answers'] += 1
            
            if result['answer_correct']:
                type_stats[scene_type]['correct_answers'] += 1
                overall_stats['correct_answers'] += 1
    
    # Calculate rates for each type
    type_metrics = {}
    for scene_type, stats in type_stats.items():
        total = stats['total']
        type_metrics[scene_type] = {
            'count': total,
            'parse_rate': round(stats['parse_success'] / total * 100, 2) if total > 0 else 0,
            'execution_rate': round(stats['execution_success'] / total * 100, 2) if total > 0 else 0,
            'answer_generation_rate': round(stats['answer_generation_success'] / total * 100, 2) if total > 0 else 0,
            'correctness_rate': round(stats['correct_answers'] / stats['evaluable_answers'] * 100, 2) if stats['evaluable_answers'] > 0 else 0,
            'error_rate': round(stats['errors'] / total * 100, 2) if total > 0 else 0,
            'evaluable_count': stats['evaluable_answers'],
            'error_count': stats['errors']
        }
    
    # Calculate overall metrics
    total = overall_stats['total_processed']
    overall_metrics = {
        'total_count': total,
        'parse_rate': round(overall_stats['parse_success'] / total * 100, 2) if total > 0 else 0,
        'execution_rate': round(overall_stats['execution_success'] / total * 100, 2) if total > 0 else 0,
        'answer_generation_rate': round(overall_stats['answer_generation_success'] / total * 100, 2) if total > 0 else 0,
        'correctness_rate': round(overall_stats['correct_answers'] / overall_stats['evaluable_answers'] * 100, 2) if overall_stats['evaluable_answers'] > 0 else 0,
        'error_rate': round(overall_stats['errors'] / total * 100, 2) if total > 0 else 0,
        'evaluable_count': overall_stats['evaluable_answers'],
        'error_count': overall_stats['errors']
    }
    
    # Save results
    output_path = Path.cwd() / args.output_file
    
    summary = {
        "processing_timestamp": datetime.now().isoformat(),
        "jsonl_source": args.jsonl_path,
        "processing_time_seconds": round(processing_time, 2),
        "avg_time_per_entry": round(processing_time/len(entries), 2),
        "num_processes_used": num_processes,
        "overall_metrics": overall_metrics,
        "type_metrics": type_metrics,
        "raw_statistics": dict(type_stats),
        "results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")
    
    # Print summary statistics
    print(f"\n=== MindCube Evaluation Results ===")
    print(f"Total entries processed: {total}")
    print(f"\n=== Overall Performance ===")
    print(f"Parse success: {overall_stats['parse_success']}/{total} ({overall_metrics['parse_rate']:.1f}%)")
    print(f"Execution success: {overall_stats['execution_success']}/{total} ({overall_metrics['execution_rate']:.1f}%)")
    print(f"Answer generation: {overall_stats['answer_generation_success']}/{total} ({overall_metrics['answer_generation_rate']:.1f}%)")
    print(f"Answer correctness: {overall_stats['correct_answers']}/{overall_stats['evaluable_answers']} ({overall_metrics['correctness_rate']:.1f}%)")
    print(f"Errors: {overall_stats['errors']}/{total} ({overall_metrics['error_rate']:.1f}%)")
    
    print(f"\n=== Statistics by Type ===")
    for scene_type, metrics in type_metrics.items():
        print(f"\n{scene_type.upper()}:")
        print(f"  Count: {metrics['count']}")
        print(f"  Parse rate: {metrics['parse_rate']:.1f}%")
        print(f"  Execution rate: {metrics['execution_rate']:.1f}%")
        print(f"  Answer generation rate: {metrics['answer_generation_rate']:.1f}%")
        print(f"  Correctness rate: {metrics['correctness_rate']:.1f}% ({type_stats[scene_type]['correct_answers']}/{metrics['evaluable_count']})")
        print(f"  Error rate: {metrics['error_rate']:.1f}% ({metrics['error_count']}/{metrics['count']})")


if __name__ == "__main__":
    # This guard is important for multiprocessing
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()








