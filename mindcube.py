
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
from contextlib import contextmanager
import time
import re
import signal
import threading
import backoff

# Add parent directory to Python path to import pySpatial_Interface
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pySpatial_Interface import Agent, Scene, pySpatial

# Rate limiting globals
last_request_time = 0
min_request_interval = 0.1  # Minimum time between requests (100ms)
request_lock = threading.Lock()

# Hard wall-clock budget (seconds) for executing a single generated program.
# Generated code is arbitrary and may infinite-loop, block on input(), or run
# away on a huge computation; without this a single bad scene hangs the whole
# evaluation. 0 disables the limit.
exec_timeout = 10


class SceneTimeout(Exception):
    """Raised when a per-scene operation exceeds its wall-clock budget."""


@contextmanager
def time_limit(seconds):
    """Abort the wrapped block with SceneTimeout after `seconds` (SIGALRM).

    Only effective in the process's main thread (true for both the sequential
    run and each multiprocessing worker). A non-positive value disables it.
    """
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise SceneTimeout(f"operation exceeded {seconds}s time limit")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def rate_limit():
    """Apply rate limiting between API requests"""
    global last_request_time
    
    with request_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < min_request_interval:
            time.sleep(min_request_interval - time_since_last)
        last_request_time = time.time()


MAX_API_TRIES = 2

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=MAX_API_TRIES,
    factor=2,
    jitter=backoff.full_jitter,
    # NOTE: backoff's `details` dict has no 'max_tries' key (only target, args,
    # kwargs, tries, elapsed, wait); referencing it here raised KeyError:
    # 'max_tries' inside the handler, masking the real error and breaking retries.
    on_backoff=lambda details: print(
        f"Retrying {details['target'].__name__} "
        f"(attempt {details['tries']}/{MAX_API_TRIES}) after {details['wait']:.1f}s..."
    ),
)
def call_agent_with_retry(agent, method_name, *args, **kwargs):
    """Call agent method with rate limiting and retry logic.

    The ``backoff.expo`` decorator drives the retry loop. For rate-limit-style
    errors we pause before re-raising so the next attempt is less likely to be
    throttled again; all errors are re-raised so backoff can retry them.
    """
    rate_limit()

    try:
        method = getattr(agent, method_name)
        return method(*args, **kwargs)
    except Exception as e:
        error_msg = str(e).lower()
        if "rate_limit" in error_msg or "429" in error_msg or "tokens per min" in error_msg:
            print(f"Rate limit hit for {method_name}, waiting before retry...")
            time.sleep(60)
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


def _normalize_option(answer: str) -> str:
    """Normalize an answer to its leading multiple-choice option letter (A-D).

    The model only ever emits a bare letter, but the dataset's ``gt_answer`` may
    carry extra formatting (e.g. "A. the cup"). Falls back to the stripped,
    uppercased string when no leading A-D token is present.
    """
    if answer is None:
        return ""
    text = str(answer).strip().upper()
    match = re.match(r"\s*([A-D])\b", text)
    return match.group(1) if match else text


def evaluate_answer_correctness(generated_answer: str, expected_answer: str) -> bool:
    """Check if generated answer matches expected answer (option-letter aware)."""
    return _normalize_option(generated_answer) == _normalize_option(expected_answer)


def _new_stats() -> Dict[str, int]:
    """Return a zeroed stats counter dict."""
    return {
        'total': 0,
        'parse_success': 0,
        'execution_success': 0,
        'answer_generation_success': 0,
        'correct_answers': 0,
        'evaluable_answers': 0,
        'errors': 0,
    }


def accumulate_stats(stats: Dict[str, int], result: Dict[str, Any]) -> None:
    """Update a single stats counter dict from one pipeline result.

    A result that errored is counted toward ``errors`` but still contributes its
    real success/correctness fields (the error path may have produced a fallback
    answer), so metrics are no longer inflated.
    """
    stats['total'] += 1
    if result.get('error'):
        stats['errors'] += 1
    if result.get('parse_success'):
        stats['parse_success'] += 1
    if result.get('execution_success'):
        stats['execution_success'] += 1
    if result.get('answer_generation_success'):
        stats['answer_generation_success'] += 1
    if result.get('expected_answer') and result.get('generated_answer'):
        stats['evaluable_answers'] += 1
        if result.get('answer_correct'):
            stats['correct_answers'] += 1


def compute_metrics(stats: Dict[str, int]) -> Dict[str, Any]:
    """Convert raw counters into percentage rates for reporting."""
    total = stats['total']
    evaluable = stats['evaluable_answers']

    def pct(numerator: int, denominator: int) -> float:
        return round(numerator / denominator * 100, 2) if denominator > 0 else 0

    return {
        'count': total,
        'parse_rate': pct(stats['parse_success'], total),
        'execution_rate': pct(stats['execution_success'], total),
        'answer_generation_rate': pct(stats['answer_generation_success'], total),
        'correctness_rate': pct(stats['correct_answers'], evaluable),
        'error_rate': pct(stats['errors'], total),
        'evaluable_count': evaluable,
        'error_count': stats['errors'],
    }


def process_scene_with_agent_wrapper(args_tuple) -> Dict[str, Any]:
    """
    Wrapper function for multiprocessing that creates its own agent instance.

    Because workers are started with the ``spawn`` method, module-level config
    set in ``main()`` is NOT inherited; we re-apply it here per child process.
    Note that rate limiting is therefore per-process, not global.

    Args:
        args_tuple: Tuple of (entry, api_key, request_interval, processed_dir, scene_exec_timeout)

    Returns:
        Dictionary containing the complete pipeline results including type information
    """
    entry, api_key, request_interval, processed_dir, scene_exec_timeout = args_tuple

    # Re-apply config that the parent set in main() (spawn does not inherit it).
    global min_request_interval, exec_timeout
    min_request_interval = request_interval
    exec_timeout = scene_exec_timeout
    pySpatial.PROCESSED_BASE_DIR = processed_dir

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

    fallback_used = False
    try:
        # Step 1: Generate code using the agent (with retry)
        generated_response = call_agent_with_retry(agent, 'generate_code', scene)
        print(f"[{scene_id}] STEP 1 generate_code: {len(generated_response or '')} chars returned")

        # Parse the response to extract code patterns
        parsed_code = agent.parse_LLM_response(scene, generated_response)
        parse_success = parsed_code is not None and parsed_code.strip() != ""

        visual_clue = None
        generated_answer = None
        answer_correct = False
        execution_success = False
        answer_generation_success = False

        if not parse_success:
            preview = (generated_response or "").strip().replace("\n", " ")[:200]
            print(f"[{scene_id}] STEP 2 parse FAILED: no ```python``` block found "
                  f"(response preview: {preview!r})")
        else:
            print(f"[{scene_id}] STEP 2 parse OK: extracted {len(parsed_code)} chars of code")

        # Step 3: Execute code to get visual clue (if parsing was successful).
        # Bounded by a wall-clock limit so a runaway generated program can't
        # stall the whole run; a timeout is treated as a failed execution.
        if parse_success:
            try:
                with time_limit(exec_timeout):
                    visual_clue = agent.execute(scene)
            except SceneTimeout as te:
                print(f"[{scene_id}] code execution timed out: {te}")
                visual_clue = "there is an error during code generation, no visual clue provided"
            execution_success = visual_clue != "there is an error during code generation, no visual clue provided"
            print(f"[{scene_id}] STEP 3 execute: success={execution_success}")

            # Step 4: Generate answer using visual clue (with retry)
            if execution_success:
                answer_response = call_agent_with_retry(agent, 'answer', scene, visual_clue)
                answer_generation_success = answer_response is not None
                print(f"[{scene_id}] STEP 4 answer: success={answer_generation_success}")

                if answer_generation_success:
                    generated_answer = answer_response.answer

                    # Step 5: Evaluate correctness
                    if expected_answer and generated_answer:
                        answer_correct = evaluate_answer_correctness(generated_answer, expected_answer)

        # --- Fallback to basic QA if pySpatial pipeline didn't produce an answer ---
        if not answer_generation_success or generated_answer is None:
            print(f"[{scene_id}] pySpatial pipeline did not produce an answer, falling back to basic QA")
            fallback_response = call_agent_with_retry(agent, 'basic_qa', scene)
            if fallback_response is not None:
                fallback_used = True
                generated_answer = fallback_response.answer
                answer_generation_success = True
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
            "fallback_used": fallback_used,
        }

        return result

    except Exception as e:
        error_msg = str(e)
        print(f"Error processing {scene_id}: {error_msg}")

        # Fallback to basic QA on complete pipeline failure
        fallback_answer = None
        fallback_correct = False
        fallback_success = False
        try:
            print(f"[{scene_id}] Pipeline error, falling back to basic QA")
            fallback_response = call_agent_with_retry(agent, 'basic_qa', scene)
            if fallback_response is not None:
                fallback_answer = fallback_response.answer
                fallback_success = True
                fallback_used = True
                if expected_answer and fallback_answer:
                    fallback_correct = evaluate_answer_correctness(fallback_answer, expected_answer)
        except Exception as fallback_e:
            print(f"[{scene_id}] Basic QA fallback also failed: {fallback_e}")

        return {
            "scene_id": scene_id,
            "scene_type": scene_type,
            "question": question,
            "images": images,
            "expected_answer": expected_answer,
            "parse_success": False,
            "execution_success": False,
            "answer_generation_success": fallback_success,
            "generated_answer": fallback_answer,
            "answer_correct": fallback_correct,
            "fallback_used": fallback_used,
            "error": error_msg,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate pySpatial Agent on MindCube dataset with type statistics")
    parser.add_argument("--jsonl_path", type=str, 
                       required=True,
                       help="Path to JSONL file containing scene information")
    parser.add_argument("--output_file", type=str,
                       default="pySpatial_mindcube_test.json",
                       help="Output file path for results")
    parser.add_argument("--max_entries", type=int, default=50,
                       help="Maximum number of entries to process")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"),
                       help="OpenAI API key (if not provided, uses OPENAI_API_KEY env var)")
    parser.add_argument("--num_processes", type=int, default=1,
                       help="Number of processes to use (default: number of CPU cores)")
    parser.add_argument("--disable_multiprocessing", action="store_true", 
                       help="Disable multiprocessing and run sequentially")
    parser.add_argument("--request_interval", type=float, default=0.1,
                       help="Minimum time between API requests in seconds (default: 0.1)")
    parser.add_argument("--filter_type", type=str, default=None,
                       choices=['among', 'around', 'rotation', 'unknown'],
                       help="Filter to only process specific scene type (among, around, rotation, or unknown)")
    parser.add_argument("--processed_dir", type=str, default=None,
                       help="Base directory for pre-processed scene data (optional)")
    parser.add_argument("--exec_timeout", type=float, default=180,
                       help="Per-scene wall-clock limit (seconds) for executing generated "
                            "code; a runaway program is treated as a failed execution. "
                            "Set to 0 to disable (default: 180)")

    args = parser.parse_args()
    
    # Update global rate limiting interval and per-scene execution timeout
    global min_request_interval, exec_timeout
    min_request_interval = args.request_interval
    exec_timeout = args.exec_timeout

    # Set the pre-processed scene base directory
    pySpatial.PROCESSED_BASE_DIR = args.processed_dir
    
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
    print(f"Filter type: {args.filter_type or 'none (processing all types)'}")
    print(f"Number of processes: {num_processes}")
    print(f"Request interval: {min_request_interval}s")
    print(f"Per-scene exec timeout: {exec_timeout or 'disabled'}s")
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

    # Filter entries by type if specified
    if args.filter_type:
        filtered_entries = []
        for entry in entries:
            images = entry.get('images', [])
            scene_type = extract_type_from_images(images)
            if scene_type == args.filter_type:
                filtered_entries.append(entry)

        print(f"Filtered to {len(filtered_entries)} entries of type '{args.filter_type}' (from {len(entries)} total)")
        entries = filtered_entries

        if len(entries) == 0:
            print(f"No entries found with type '{args.filter_type}'. Exiting.")
            return

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

        # Prepare arguments for multiprocessing. request_interval and
        # processed_dir are passed explicitly because spawn workers do not
        # inherit the globals set above.
        args_list = [
            (entry, args.api_key, min_request_interval, pySpatial.PROCESSED_BASE_DIR, exec_timeout)
            for entry in entries
        ]

        pool = Pool(processes=num_processes, maxtasksperchild=4)
        async_result = pool.map_async(process_scene_with_agent_wrapper, args_list)
        results = async_result.get(timeout=3600)
        pool.terminate()
        pool.join()
    
    end_time = time.time()
    processing_time = end_time - start_time
    avg_time_per_entry = processing_time / len(entries) if entries else 0.0
    print(f"\n✓ Processing completed in {processing_time:.2f} seconds")
    print(f"Average time per entry: {avg_time_per_entry:.2f} seconds")

    # Accumulate per-type statistics, then derive overall by summing across types.
    type_stats = defaultdict(_new_stats)
    for result in results:
        accumulate_stats(type_stats[result['scene_type']], result)

    overall_stats = _new_stats()
    for stats in type_stats.values():
        for key in overall_stats:
            overall_stats[key] += stats[key]

    type_metrics = {scene_type: compute_metrics(stats) for scene_type, stats in type_stats.items()}

    overall_metrics = compute_metrics(overall_stats)
    overall_metrics['total_count'] = overall_metrics.pop('count')

    # Save results
    output_path = Path.cwd() / args.output_file
    
    summary = {
        "processing_timestamp": datetime.now().isoformat(),
        "jsonl_source": args.jsonl_path,
        "processing_time_seconds": round(processing_time, 2),
        "avg_time_per_entry": round(avg_time_per_entry, 2),
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
    total = overall_stats['total']
    print(f"\n=== MindCube Evaluation Results ===")
    print(f"Total entries processed: {total}")
    print(f"\n=== Overall Performance ===")
    print(f"Parse success: {overall_stats['parse_success']}/{total} ({overall_metrics['parse_rate']:.1f}%)")
    print(f"Execution success: {overall_stats['execution_success']}/{total} ({overall_metrics['execution_rate']:.1f}%)")
    print(f"Answer generation: {overall_stats['answer_generation_success']}/{total} ({overall_metrics['answer_generation_rate']:.1f}%)")
    print(f"Answer correctness: {overall_stats['correct_answers']}/{overall_stats['evaluable_answers']} ({overall_metrics['correctness_rate']:.1f}%)")

    
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
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user, shutting down...")
        sys.exit(1)








