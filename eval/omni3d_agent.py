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
from PIL import Image
import argparse
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import threading
import backoff
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal
from pySpatial_Interface import agent, Scene

# Rate limiting globals
last_request_time = 0
min_request_interval = 0.1  # Minimum time between requests (100ms)
request_lock = threading.Lock()


# Pydantic models for structured output
class FloatAnswer(BaseModel):
    reasoning: str
    answer: float


class IntAnswer(BaseModel):
    reasoning: str
    answer: int


class TrueFalseAnswer(BaseModel):
    reasoning: str
    answer: Literal["yes", "no"]


class MultiChoiceAnswer(BaseModel):
    reasoning: str
    answer: str


# wrap the quesiton into a scene object

# solve the problem with agent 
def sovle(input: Scene):
    # initialize the agent
    agent = Agent(api_key=os.getenv('OPENAI_API_KEY'))
    agent.generate_code(input) # noted that the visual clue will be saved in the scene object
    agent.execute(input)
    agent.answer(input)
    return agent.answer



def rate_limit():
    """Apply rate limiting between API requests"""
    global last_request_time

    with request_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < min_request_interval:
            time.sleep(min_request_interval - time_since_last)
        last_request_time = time.time()


def encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """Encode image to base64 for OpenAI API.

    Returns:
        tuple: (base64_string, mime_type)
    """
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    # Detect image format using PIL
    from io import BytesIO
    img = Image.open(BytesIO(image_data))
    format_lower = img.format.lower()

    # Map PIL format to MIME type
    mime_type_map = {
        'jpeg': 'image/jpeg',
        'jpg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'bmp': 'image/bmp',
        'webp': 'image/webp'
    }

    mime_type = mime_type_map.get(format_lower, 'image/jpeg')  # Default to jpeg
    base64_string = base64.b64encode(image_data).decode('utf-8')

    return base64_string, mime_type


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    factor=2,
    jitter=backoff.full_jitter,
    on_backoff=lambda details: print(f"Retrying API call (attempt {details['tries']}/{details['max_tries']})...")
)
def query_chatgpt_vision_with_retry(image_path: str, question: str, api_key: str, model: str = "gpt-4o", question_type: str = "str") -> tuple[str, str]:
    """Query ChatGPT Vision API with retry logic and rate limiting."""
    rate_limit()

    try:
        return query_chatgpt_vision(image_path, question, api_key, model, question_type)
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
            print(f"Circumstance 3: We are handling this error: {e}")
            raise


def query_chatgpt_vision(image_path: str, question: str, api_key: str, model: str = "gpt-4o", question_type: str = "str") -> tuple[str, str]:
    """Query ChatGPT Vision API with an image and question using structured output.

    Returns:
        tuple: (answer, reasoning) where answer is the parsed response and reasoning is the model's reasoning
    """

    # Encode the image
    base64_image, mime_type = encode_image_to_base64(image_path)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Create the message with image and text
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": get_type_specific_prompt(question, question_type)
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    # Select the appropriate response model based on question type
    if question_type == "float":
        response_model = FloatAnswer
    elif question_type == "int":
        response_model = IntAnswer
    elif question_type == "true_or_false":
        response_model = TrueFalseAnswer
    elif question_type == "multi_choice":
        response_model = MultiChoiceAnswer
    else:
        # Default to string output for unknown types
        response_model = None

    try:
        if response_model:
            # Use structured output for specific question types
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_model,
                max_tokens=1000,
                temperature=0.1
            )

            parsed_response = response.choices[0].message.parsed
            if not parsed_response:
                raise Exception("Failed to parse structured response")

            # Return both answer and reasoning
            return str(parsed_response.answer), parsed_response.reasoning
        else:
            # Fallback to regular completion for non-structured types
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0.1
            )

            content = response.choices[0].message.content.strip()
            if not content:
                raise Exception("Empty response from API")

            # For non-structured responses, return content as both answer and reasoning
            return content, "No structured reasoning available"

    except Exception as e:
        # Re-raise to allow backoff to work
        raise Exception(f"OpenAI API error: {str(e)}")


def get_refined_question_type(question: str, answer: Union[str, int, float], answer_type: str) -> str:
    """
    Determine refined question type:
    - float: numerical ratios/measurements
    - int: counting questions
    - true_or_false: yes/no questions
    - multi_choice: multiple choice questions
    """
    if answer_type in ['float', 'int']:
        return answer_type
    elif answer_type == 'str':
        answer_str = str(answer).strip().lower()
        if answer_str in ['yes', 'no']:
            return 'true_or_false'
        else:
            return 'multi_choice'
    else:
        return answer_type


def get_type_specific_prompt(question: str, question_type: str) -> str:
    """Generate type-specific prompts for different question types with structured output."""

    base_prompt = f"""You are an expert at spatial reasoning and 3D scene understanding.

    Question: {question}

    Please analyze the image carefully and provide a precise answer with your reasoning."""

    if question_type == "float":
        # print(f"Float question: {question}")
        specific_guidance = """

    For this numerical ratio/measurement question:
    - Carefully observe the spatial relationships and dimensions
    - Calculate the ratio or measurement requested
    - Provide your reasoning process and the numerical value
    - Be as precise as possible based on visual estimation

    IMPORTANT: You must provide a numerical answer. If you're uncertain, provide your best estimate based on visual analysis.
    If you cannot determine the exact value, estimate based on typical proportions (default to 2.0 if completely unsure).

    Provide both your reasoning and the numerical answer."""

    elif question_type == "int":
        specific_guidance = """

    For this counting question:
    - Carefully count all relevant objects in the image
    - Make sure to identify all instances, including partially visible ones
    - Provide your counting process and the final integer count

    IMPORTANT: You must provide an integer count. If some objects are partially obscured, make your best estimate.

    Provide both your reasoning and the integer count."""

    elif question_type == "true_or_false":
        specific_guidance = """

    For this yes/no question:
    - Analyze the spatial scenario described in the question
    - Consider visibility, occlusion, and spatial relationships
    - Provide your reasoning and answer with "yes" or "no"

    IMPORTANT: You must choose either "yes" or "no". If uncertain, make your best judgment based on the visual evidence.

    Provide both your reasoning and your yes/no answer."""

    elif question_type == "multi_choice":
        specific_guidance = """

    For this multiple choice question:
    - Look for the options mentioned in the question (usually in format "Options: {option1, option2}")
    - Analyze the spatial relationships to determine the correct option
    - Provide your reasoning and the exact option text

    IMPORTANT: Your answer must be exactly one of the provided options.
    For example, if the question is "Which object is closer: the sofa or the coffee table? Options: {sofa, coffee table}"
    Your answer should be exactly "sofa" or "coffee table" (the exact option text).

    Provide both your reasoning and the exact option text."""

    else:
        # Default guidance for unknown types
        specific_guidance = """

    Please provide your reasoning and answer following these guidelines:
    - For numerical answers (ratios, measurements), provide only the number
    - For yes/no questions, answer only "yes" or "no"
    - For multiple choice questions, provide only the exact option text
    - For counting questions, provide only the integer count

    Provide both your reasoning process and your final answer."""

    return base_prompt + specific_guidance


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
        # we use a confidence threshold
        confidence_thresholds = [0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
        if ground_truth == 0:
            return abs(predicted) < 1e-6
        confidence_score = 0
        for thres in confidence_thresholds:
            if abs(predicted - ground_truth) / abs(ground_truth) <= thres:
                confidence_score += 1
        return confidence_score  / 10


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
        # Get refined type for consistency
        try:
            refined_type = get_refined_question_type(item['question'], item['answer'], item['answer_type'])
        except:
            refined_type = item['answer_type']

        return {
            "question_index": item['question_index'],
            "image_filename": item['image_filename'],
            "question": item['question'],
            "ground_truth": item['answer'],
            "predicted": None,
            "raw_answer": "",
            "reasoning": "Error: Image not found",
            "answer_type": item['answer_type'],
            "refined_answer_type": refined_type,
            "correct": False,
            "error": f"Image not found: {image_path}"
        }

    try:
        # Determine refined question type
        refined_type = get_refined_question_type(item['question'], item['answer'], item['answer_type'])

        # Query the model with retry logic, passing the refined type
        raw_answer, reasoning = query_chatgpt_vision_with_retry(image_path, item['question'], api_key, model, refined_type)

        # For structured output, the raw_answer is just the answer value
        # We need to parse it appropriately
        predicted = parse_answer(raw_answer, item['answer_type'])
        ground_truth = item['answer']

        # Evaluate
        is_correct = evaluate_answer(predicted, ground_truth, item['answer_type'])

        # Return result with both original and refined types, including reasoning
        return {
            "question_index": item['question_index'],
            "image_filename": item['image_filename'],
            "question": item['question'],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "raw_answer": raw_answer,
            "reasoning": reasoning,
            "answer_type": item['answer_type'],
            "refined_answer_type": refined_type,
            "correct": is_correct
        }

    except Exception as e:
        # Still try to get the refined type for consistency
        try:
            refined_type = get_refined_question_type(item['question'], item['answer'], item['answer_type'])
        except:
            refined_type = item['answer_type']

        return {
            "question_index": item['question_index'],
            "image_filename": item['image_filename'],
            "question": item['question'],
            "ground_truth": item['answer'],
            "predicted": None,
            "raw_answer": "",
            "reasoning": f"Error during processing: {str(e)}",
            "answer_type": item['answer_type'],
            "refined_answer_type": refined_type,
            "correct": False,
            "error": str(e)
        }


def evaluate_omni3d(
    annotations_path: str,
    images_dir: str,
    api_key: str,
    output_path: str = "omni3d_results_refined.json",
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
    # Default to single process to avoid rate limit violations in multiprocessing
    if disable_multiprocessing:
        num_processes = 1
    elif num_processes is None:
        num_processes = 1  # Safe default
    else:
        # User explicitly set num_processes, respect their choice but warn
        if num_processes > 1:
            print(f"WARNING: Using {num_processes} processes may cause rate limit violations.")
            print("Consider using --num-processes 1 or --disable-multiprocessing for API rate limiting.")

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

    # Calculate final metrics for both original and refined types
    # Original metrics (for compatibility)
    correct_by_type = {"float": 0, "int": 0, "str": 0}
    total_by_type = {"float": 0, "int": 0, "str": 0}

    # Refined metrics
    refined_correct_by_type = {"float": 0, "int": 0, "true_or_false": 0, "multi_choice": 0}
    refined_total_by_type = {"float": 0, "int": 0, "true_or_false": 0, "multi_choice": 0}
    error_count = 0

    for result in results:
        if result.get('error'):
            error_count += 1
            continue

        # Original type counting
        answer_type = result['answer_type']
        total_by_type[answer_type] += 1

        # For float questions, 'correct' is a confidence score (0.0-1.0)
        # For other questions, 'correct' is a boolean
        if answer_type == "float":
            correct_by_type[answer_type] += result['correct']
        else:
            if result['correct']:
                correct_by_type[answer_type] += 1

        # Refined type counting
        refined_type = result.get('refined_answer_type', answer_type)
        if refined_type in refined_total_by_type:
            refined_total_by_type[refined_type] += 1
            if refined_type == "float":
                refined_correct_by_type[refined_type] += result['correct']
            else:
                if result['correct']:
                    refined_correct_by_type[refined_type] += 1

    # Calculate original metrics (for compatibility)
    metrics = {}
    for answer_type in ["float", "int", "str"]:
        if total_by_type[answer_type] > 0:
            metrics[f"{answer_type}_accuracy"] = correct_by_type[answer_type] / total_by_type[answer_type]
            metrics[f"{answer_type}_count"] = total_by_type[answer_type]
        else:
            metrics[f"{answer_type}_accuracy"] = 0.0
            metrics[f"{answer_type}_count"] = 0

    # Calculate refined metrics
    refined_metrics = {}
    for answer_type in ["float", "int", "true_or_false", "multi_choice"]:
        if refined_total_by_type[answer_type] > 0:
            refined_metrics[f"{answer_type}_accuracy"] = refined_correct_by_type[answer_type] / refined_total_by_type[answer_type]
            refined_metrics[f"{answer_type}_count"] = refined_total_by_type[answer_type]
            refined_metrics[f"{answer_type}_correct"] = refined_correct_by_type[answer_type]
        else:
            refined_metrics[f"{answer_type}_accuracy"] = 0.0
            refined_metrics[f"{answer_type}_count"] = 0
            refined_metrics[f"{answer_type}_correct"] = 0

    total_valid = sum(total_by_type.values())
    refined_total_valid = sum(refined_total_by_type.values())

    if total_valid > 0:
        metrics["overall_accuracy"] = sum(correct_by_type.values()) / total_valid
    else:
        metrics["overall_accuracy"] = 0.0

    if refined_total_valid > 0:
        refined_metrics["overall_accuracy"] = sum(refined_correct_by_type.values()) / refined_total_valid
    else:
        refined_metrics["overall_accuracy"] = 0.0

    metrics["total_questions"] = total_valid
    metrics["error_count"] = error_count
    refined_metrics["total_questions"] = refined_total_valid
    refined_metrics["error_count"] = error_count
    metrics["processing_time_seconds"] = round(processing_time, 2)
    refined_metrics["processing_time_seconds"] = round(processing_time, 2)

    # Save results with both original and refined metrics
    final_results = {
        "metrics": metrics,  # Original metrics for compatibility
        "refined_metrics": refined_metrics,  # New refined metrics
        "results": results,
        "model": model,
        "start_idx": start_idx,
        "num_processes_used": num_processes,
        "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {model}")
    print(f"Total Valid Questions: {metrics['total_questions']}")
    print(f"Errors: {metrics['error_count']}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.3f}")

    print("\nOriginal Categories:")
    for answer_type in ["float", "int", "str"]:
        acc = metrics[f"{answer_type}_accuracy"]
        count = metrics[f"{answer_type}_count"]
        print(f"  {answer_type.upper()}: {acc:.3f} ({count} questions)")

    print("\nRefined Categories:")
    for answer_type in ["float", "int", "true_or_false", "multi_choice"]:
        acc = refined_metrics[f"{answer_type}_accuracy"]
        count = refined_metrics[f"{answer_type}_count"]
        correct = refined_metrics[f"{answer_type}_correct"]
        print(f"  {answer_type.replace('_', '-').upper()}: {acc:.3f} ({correct}/{count})")

    print(f"\nRefined Overall Accuracy: {refined_metrics['overall_accuracy']:.3f}")
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Omni3D dataset")
    parser.add_argument("--annotations", default="/data/Datasets/MindCube/data/omni3D/annotations.json",
                       help="Path to annotations.json")
    parser.add_argument("--images", default="/data/Datasets/MindCube/data/omni3D/images",
                       help="Path to images directory")
    parser.add_argument("--api-key", default=os.getenv('OPENAI_API_KEY'), help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o", help="Model to use")
    parser.add_argument("--output", default="omni3d_results_structured_v1.json", help="Output file")
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