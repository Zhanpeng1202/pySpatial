#!/usr/bin/env python3
"""
Omni3D Spatial Reasoning Evaluation Script

This script evaluates a model's performance on the Omni3D dataset, which contains
spatial reasoning questions about indoor scenes from ARKitScenes.

Dataset structure:
- Questions involve spatial relationships, size comparisons, object positioning
- Answer types: float (ratios, measurements), str (yes/no, multiple choice), int (counts)
- Images are from ARKitScenes indoor environments
"""

import json
import os
import base64
from pathlib import Path
from typing import Dict, List, Any, Union
import requests
from PIL import Image
import argparse
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import threading
import backoff

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


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 for OpenAI API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    factor=2,
    jitter=backoff.full_jitter,
    on_backoff=lambda details: print(f"Retrying API call (attempt {details['tries']}/{details['max_tries']})...")
)
def query_chatgpt_vision_with_retry(image_path: str, question: str, api_key: str, model: str = "gpt-4o") -> str:
    """Query ChatGPT Vision API with retry logic and rate limiting."""
    rate_limit()

    try:
        return query_chatgpt_vision(image_path, question, api_key, model)
    except Exception as e:
        error_msg = str(e)

        # Handle rate limit specifically
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            print(f"Rate limit hit, waiting before retry...")
            time.sleep(60)
            raise
        elif "tokens per min" in error_msg.lower():
            print(f"Token rate limit hit, waiting before retry...")
            time.sleep(60)
            raise
        else:
            # For other errors, re-raise without modification
            raise


def query_chatgpt_vision(image_path: str, question: str, api_key: str, model: str = "gpt-4o") -> str:
    """Query ChatGPT Vision API with an image and question."""

    # Encode the image
    base64_image = encode_image_to_base64(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are an expert at spatial reasoning and 3D scene understanding.

Question: {question}

Please analyze the image carefully and provide a precise answer.

Guidelines:
- For numerical answers (ratios, measurements), provide only the number (e.g., "2.5" not "2.5 meters")
- For yes/no questions, answer only "yes" or "no"
- For multiple choice questions, provide only the exact option text
- For counting questions, provide only the integer count

if you are not sure about the answer, just give you best guess.

Answer:"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100,
        "temperature": 0.1
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        return result['choices'][0]['message']['content'].strip()

    except Exception as e:
        print(f"Error querying API: {e}")
        return ""


def parse_answer(raw_answer: str, answer_type: str) -> Union[float, int, str]:
    """Parse the raw answer based on expected type."""
    raw_answer = raw_answer.strip().lower()

    if answer_type == "float":
        # Extract numerical value
        import re
        numbers = re.findall(r'-?\d+\.?\d*', raw_answer)
        if numbers:
            return float(numbers[0])
        return 0.0

    elif answer_type == "int":
        # Extract integer value
        import re
        numbers = re.findall(r'\d+', raw_answer)
        if numbers:
            return int(numbers[0])
        return 0

    elif answer_type == "str":
        # Handle yes/no and multiple choice
        if "yes" in raw_answer:
            return "yes"
        elif "no" in raw_answer:
            return "no"
        else:
            # For multiple choice, return the cleaned answer
            return raw_answer

    return raw_answer


def evaluate_answer(predicted: Union[float, int, str], ground_truth: Union[float, int, str], answer_type: str) -> bool:
    """Evaluate if the predicted answer matches ground truth."""

    if answer_type == "float":
        # Use relative tolerance for float comparison
        if ground_truth == 0:
            return abs(predicted) < 1e-6
        return abs(predicted - ground_truth) / abs(ground_truth) < 0.1  # 10% tolerance

    elif answer_type == "int":
        return predicted == ground_truth

    elif answer_type == "str":
        return str(predicted).strip().lower() == str(ground_truth).strip().lower()

    return False


def load_dataset(annotations_path: str) -> List[Dict[str, Any]]:
    """Load the Omni3D dataset."""
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    return data['questions']


def process_question_wrapper(args_tuple) -> Dict[str, Any]:
    """
    Wrapper function for multiprocessing that handles a single question.

    Args:
        args_tuple: Tuple of (item, images_dir, api_key, model)

    Returns:
        Dictionary containing the evaluation result
    """
    item, images_dir, api_key, model = args_tuple

    # Construct image path
    image_path = os.path.join(images_dir, item['image_filename'])

    if not os.path.exists(image_path):
        return {
            "question_index": item['question_index'],
            "image_filename": item['image_filename'],
            "question": item['question'],
            "ground_truth": item['answer'],
            "predicted": None,
            "raw_answer": "",
            "answer_type": item['answer_type'],
            "correct": False,
            "error": f"Image not found: {image_path}"
        }

    try:
        # Query the model with retry logic
        raw_answer = query_chatgpt_vision_with_retry(image_path, item['question'], api_key, model)

        # Parse the answer
        predicted = parse_answer(raw_answer, item['answer_type'])
        ground_truth = item['answer']

        # Evaluate
        is_correct = evaluate_answer(predicted, ground_truth, item['answer_type'])

        # Return result
        return {
            "question_index": item['question_index'],
            "image_filename": item['image_filename'],
            "question": item['question'],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "raw_answer": raw_answer,
            "answer_type": item['answer_type'],
            "correct": is_correct
        }

    except Exception as e:
        return {
            "question_index": item['question_index'],
            "image_filename": item['image_filename'],
            "question": item['question'],
            "ground_truth": item['answer'],
            "predicted": None,
            "raw_answer": "",
            "answer_type": item['answer_type'],
            "correct": False,
            "error": str(e)
        }


def evaluate_omni3d(
    annotations_path: str,
    images_dir: str,
    api_key: str,
    output_path: str = "omni3d_results.json",
    model: str = "gpt-4o",
    max_samples: int = None,
    start_idx: int = 0,
    num_processes: int = None,
    disable_multiprocessing: bool = False
):
    """Evaluate on Omni3D dataset."""

    print(f"Loading dataset from {annotations_path}...")
    questions = load_dataset(annotations_path)

    if max_samples:
        questions = questions[start_idx:start_idx + max_samples]
    else:
        questions = questions[start_idx:]

    # Determine number of processes
    if disable_multiprocessing:
        num_processes = 1
    else:
        num_processes = num_processes or cpu_count()

    print(f"Evaluating {len(questions)} questions starting from index {start_idx}...")
    print(f"Using {num_processes} processes")

    start_time = time.time()

    if num_processes == 1 or disable_multiprocessing:
        # Sequential processing
        print("Running sequentially...")
        results = []
        for i, item in enumerate(tqdm(questions, desc="Evaluating")):
            print(f"Processing question {i+1}/{len(questions)}: {item.get('question_index', 'unknown')}")
            result = process_question_wrapper((item, images_dir, api_key, model))
            results.append(result)
    else:
        # Multiprocessing
        print(f"Running with {num_processes} processes...")

        # Prepare arguments for multiprocessing
        args_list = [(item, images_dir, api_key, model) for item in questions]

        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_question_wrapper, args_list),
                total=len(questions),
                desc="Evaluating"
            ))

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\n✓ Processing completed in {processing_time:.2f} seconds")
    print(f"Average time per question: {processing_time/len(questions):.2f} seconds")

    # Calculate final metrics
    correct_by_type = {"float": 0, "int": 0, "str": 0}
    total_by_type = {"float": 0, "int": 0, "str": 0}
    error_count = 0

    for result in results:
        if result.get('error'):
            error_count += 1
            continue

        answer_type = result['answer_type']
        total_by_type[answer_type] += 1
        if result['correct']:
            correct_by_type[answer_type] += 1

    # Calculate metrics
    metrics = {}
    for answer_type in ["float", "int", "str"]:
        if total_by_type[answer_type] > 0:
            metrics[f"{answer_type}_accuracy"] = correct_by_type[answer_type] / total_by_type[answer_type]
            metrics[f"{answer_type}_count"] = total_by_type[answer_type]
        else:
            metrics[f"{answer_type}_accuracy"] = 0.0
            metrics[f"{answer_type}_count"] = 0

    total_valid = sum(total_by_type.values())
    if total_valid > 0:
        metrics["overall_accuracy"] = sum(correct_by_type.values()) / total_valid
    else:
        metrics["overall_accuracy"] = 0.0

    metrics["total_questions"] = total_valid
    metrics["error_count"] = error_count
    metrics["processing_time_seconds"] = round(processing_time, 2)

    # Save results
    final_results = {
        "metrics": metrics,
        "results": results,
        "model": model,
        "start_idx": start_idx,
        "num_processes_used": num_processes,
        "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {model}")
    print(f"Total Valid Questions: {metrics['total_questions']}")
    print(f"Errors: {metrics['error_count']}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.3f}")
    print("\nBy Answer Type:")
    for answer_type in ["float", "int", "str"]:
        acc = metrics[f"{answer_type}_accuracy"]
        count = metrics[f"{answer_type}_count"]
        print(f"  {answer_type.upper()}: {acc:.3f} ({count} questions)")
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Omni3D dataset")
    parser.add_argument("--annotations", default="/data/Datasets/MindCube/data/omni3D/annotations.json",
                       help="Path to annotations.json")
    parser.add_argument("--images", default="/data/Datasets/MindCube/data/omni3D/images",
                       help="Path to images directory")
    parser.add_argument("--api-key", default=os.getenv('OPENAI_API_KEY'), help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--output", default="omni3d_results.json", help="Output file")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting index for evaluation")
    parser.add_argument("--num-processes", type=int, default=16, help="Number of processes to use")
    parser.add_argument("--disable-multiprocessing", action="store_true", help="Disable multiprocessing")
    parser.add_argument("--request-interval", type=float, default=0.1, help="Minimum time between requests")

    args = parser.parse_args()

    # Update global rate limiting interval
    global min_request_interval
    min_request_interval = args.request_interval

    # Verify paths exist
    if not os.path.exists(args.annotations):
        print(f"Error: Annotations file not found: {args.annotations}")
        return

    if not os.path.exists(args.images):
        print(f"Error: Images directory not found: {args.images}")
        return

    evaluate_omni3d(
        annotations_path=args.annotations,
        images_dir=args.images,
        api_key=args.api_key,
        output_path=args.output,
        model=args.model,
        max_samples=args.max_samples,
        start_idx=args.start_idx,
        num_processes=args.num_processes,
        disable_multiprocessing=args.disable_multiprocessing
    )


if __name__ == "__main__":
    # This guard is important for multiprocessing
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()