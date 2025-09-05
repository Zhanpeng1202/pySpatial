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
import shutil
from pathlib import Path
from typing import Dict, Any, List, Union
from datetime import datetime

# Add parent directory to Python path to import pySpatial_Interface
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pySpatial_Interface import Agent, Scene


def save_visual_clue(visual_clue: Union[str, List[str], None], scene_id: str) -> Union[str, List[str], None]:
    """
    Save visual clues to eval/visualClue directory if they are images.
    
    Args:
        visual_clue: The visual clue (string, image path, PIL Image, or list)
        scene_id: Scene identifier for naming saved files
        
    Returns:
        The saved file paths or original string (JSON serializable)
    """
    visual_clue_dir = Path("eval/visualClue")
    visual_clue_dir.mkdir(exist_ok=True)
    
    if isinstance(visual_clue, str):
        # Check if it's a file path to an image
        if os.path.exists(visual_clue) and visual_clue.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # It's an image file, copy it
            file_ext = Path(visual_clue).suffix
            saved_path = visual_clue_dir / f"{scene_id}_visual_clue{file_ext}"
            shutil.copy2(visual_clue, saved_path)
            return str(saved_path)
        else:
            # It's a text string, return as-is
            return visual_clue
    
    elif hasattr(visual_clue, 'save'):
        # PIL Image object or similar
        saved_path = visual_clue_dir / f"{scene_id}_visual_clue.png"
        visual_clue.save(saved_path)
        return str(saved_path)
    
    elif 'open3d' in str(type(visual_clue)) and hasattr(visual_clue, 'width') and hasattr(visual_clue, 'height'):
        # Open3D Image object - convert to PIL and save
        print(f"  Debug - save_visual_clue: Detected Open3D Image, converting...")
        import numpy as np
        from PIL import Image
        
        try:
            # Convert Open3D image to numpy array
            img_array = np.asarray(visual_clue)
            print(f"  Debug - save_visual_clue: Numpy conversion shape={img_array.shape}, dtype={img_array.dtype}")
            
            # Convert to PIL Image
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)
                print(f"  Debug - save_visual_clue: Normalized to uint8")
            
            pil_image = Image.fromarray(img_array)
            saved_path = visual_clue_dir / f"{scene_id}_visual_clue.png"
            pil_image.save(saved_path)
            print(f"  Debug - save_visual_clue: Successfully saved PNG to {saved_path}")
            return str(saved_path)
        except Exception as e:
            print(f"  Debug - save_visual_clue: Open3D conversion failed: {e}")
            # Fall back to text file
            text_content = str(visual_clue)
            saved_path = visual_clue_dir / f"{scene_id}_visual_clue.txt"
            with open(saved_path, 'w') as f:
                f.write(text_content)
            print(f"  Debug - save_visual_clue: Saved as text file instead")
            return str(saved_path)
    
    elif isinstance(visual_clue, list):
        # List of images, save each one
        saved_paths = []
        
        for i, item in enumerate(visual_clue):
            if isinstance(item, str):
                if os.path.exists(item) and item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    file_ext = Path(item).suffix
                    saved_path = visual_clue_dir / f"{scene_id}_visual_clue_{i}{file_ext}"
                    shutil.copy2(item, saved_path)
                    saved_paths.append(str(saved_path))
                else:
                    saved_paths.append(item)
            elif hasattr(item, 'save'):
                # PIL Image object
                saved_path = visual_clue_dir / f"{scene_id}_visual_clue_{i}.png"
                item.save(saved_path)
                saved_paths.append(str(saved_path))
            elif 'open3d' in str(type(item)) and hasattr(item, 'width') and hasattr(item, 'height'):
                # Open3D Image object
                import numpy as np
                from PIL import Image
                
                # Convert Open3D image to numpy array
                img_array = np.asarray(item)
                
                # Convert to PIL Image
                if img_array.dtype != np.uint8:
                    img_array = (img_array * 255).astype(np.uint8)
                
                pil_image = Image.fromarray(img_array)
                saved_path = visual_clue_dir / f"{scene_id}_visual_clue_{i}.png"
                pil_image.save(saved_path)
                saved_paths.append(str(saved_path))
            else:
                # Convert non-serializable objects to string
                saved_paths.append(str(item))
        return saved_paths
    
    elif visual_clue is None:
        return None
    
    else:
        # For any other non-serializable type, convert to string and save as text file
        text_content = str(visual_clue)
        saved_path = visual_clue_dir / f"{scene_id}_visual_clue.txt"
        with open(saved_path, 'w') as f:
            f.write(text_content)
        return str(saved_path)


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
    """
    Simple evaluation of answer correctness.
    This can be extended with more sophisticated matching logic.
    
    Args:
        generated_answer: The answer generated by the model
        expected_answer: The expected correct answer
        
    Returns:
        True if answers match, False otherwise
    """
    if not generated_answer or not expected_answer:
        return False
    
    # Simple string matching (can be made more sophisticated)
    generated_clean = generated_answer.strip().lower()
    expected_clean = expected_answer.strip().lower()
    
    return generated_clean == expected_clean


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
    scene_id = entry['id']
    question = entry.get('question', '')
    images = entry.get('images', [])
    expected_answer = entry.get('answer', '')  # Get expected answer if available
    
    # Create Scene object (using dummy images since we don't have actual image paths)
    scene = Scene(images if images else [], question, scene_id=scene_id)
    
    # Step 1: Generate code using the agent
    generated_response = agent.generate_code(scene)
    
    # Parse the response to extract code patterns
    parsed_code = agent.parse_LLM_response(scene, generated_response)
    parse_success = parsed_code is not None and parsed_code.strip() != ""
    
    visual_clue = None
    visual_clue_saved_path = None
    generated_answer = None
    answer_correct = False
    execution_success = False
    answer_generation_success = False
    
    # Step 2: Execute code to get visual clue (if parsing was successful)
    if parse_success:
        visual_clue = agent.execute(scene)
        execution_success = visual_clue != "there is an error during code generation, no visual clue provided"
        
        if not execution_success:
            print(f"  Execution failed for scene {scene_id}: {visual_clue}")
            if hasattr(agent, 'last_execution_error'):
                print(f"  Additional error info: {agent.last_execution_error}")
        
        if execution_success:
            # Debug: print visual clue type and content
            print(f"  Debug - Visual clue type: {type(visual_clue)}")
            print(f"  Debug - Has width attr: {hasattr(visual_clue, 'width')}")
            print(f"  Debug - Has height attr: {hasattr(visual_clue, 'height')}")
            print(f"  Debug - open3d in type: {'open3d' in str(type(visual_clue))}")
            
            if hasattr(visual_clue, 'width') and hasattr(visual_clue, 'height'):
                print(f"  Debug - Image dimensions: {visual_clue.width}x{visual_clue.height}")
                # Try to convert Open3D image to numpy to verify it works
                import numpy as np
                img_array = np.asarray(visual_clue)
                print(f"  Debug - Numpy conversion successful: shape={img_array.shape}, dtype={img_array.dtype}")
            else:
                print(f"  Debug - Visual clue content (first 100 chars): {str(visual_clue)[:100]}")
            
            # Save visual clue if it's an image
            visual_clue_saved_path = save_visual_clue(visual_clue, scene_id)
            
            # Step 3: Generate answer using visual clue
            generated_answer = agent.answer(scene, visual_clue)
            answer_generation_success = generated_answer is not None
            
            # Step 4: Evaluate correctness
            if expected_answer and generated_answer:
                answer_correct = evaluate_answer_correctness(generated_answer, expected_answer)
        
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
            "visual_clue_saved_path": visual_clue_saved_path,
            "answer_generation_success": answer_generation_success,
            "generated_answer": generated_answer,
            "answer_correct": answer_correct,
            "status": "success"
        }
        
        status_msg = f"Scene {scene_id}: Code✓" if parse_success else f"Scene {scene_id}: Code✗"
        if execution_success:
            status_msg += ", Exec✓"
        else:
            exec_error = result.get("execution_error", "Unknown error")
            # Truncate long error messages for status display
            if exec_error and len(exec_error) > 100:
                exec_error_short = exec_error[:97] + "..."
            else:
                exec_error_short = exec_error
            status_msg += f", Exec✗ ({exec_error_short})" if exec_error and exec_error != "there is an error during code generation, no visual clue provided" else ", Exec✗"
        if answer_generation_success:
            status_msg += ", Answer✓"
        else:
            status_msg += ", Answer✗"
        if answer_correct:
            status_msg += ", Correct✓"
        elif expected_answer:
            status_msg += ", Correct✗"
        
    print(f"✓ {status_msg}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Test pySpatial Agent code generation on MindCube scenes")
    parser.add_argument("--jsonl_path", type=str, 
                       default="/data/Datasets/MindCube/data/raw/MindCube_tinybench.jsonl",
                       help="Path to JSONL file containing scene information")
    parser.add_argument("--output_file", type=str,
                       default="agent_code_generation_results_v3.json",
                       help="Output file path for results")
    parser.add_argument("--max_entries", type=int, default=1,
                       help="Maximum number of entries to process (for testing)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API key (if not provided, uses OPENAI_API_KEY env var)")
    parser.add_argument("--translation_only", action="store_true", default=False,
                       help="Only process translation questions (default: process all questions)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl_path):
        raise ValueError(f"JSONL file not found: {args.jsonl_path}")
    
    # Initialize Agent
    agent = Agent(api_key=args.api_key)
    print("✓ Agent initialized successfully")
    
    print(f"Processing JSONL file: {args.jsonl_path}")
    print(f"Output file: {args.output_file}")
    print(f"Max entries: {args.max_entries or 'unlimited'}")
    print("="*60)
    
    # Process JSONL entries
    processed_count = 0
    successful_count = 0
    failed_count = 0
    parse_success_count = 0
    execution_success_count = 0
    answer_success_count = 0
    correct_answer_count = 0
    evaluable_answer_count = 0  # Answers where expected answer is available
    results = []
    
    with open(args.jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if args.max_entries and processed_count >= args.max_entries:
                print(f"Reached maximum entries limit: {args.max_entries}")
                break
            
            entry = json.loads(line.strip())
            
            # Check if this is a translation question (if filtering is enabled)
            if args.translation_only:
                question = entry.get('question', '')
                is_translation_question = question.startswith("Based on these two views showing the same scene: in which direction did I move from the first view to the second view")
                
                # Only process translation questions when filtering is enabled
                if not is_translation_question:
                    continue
            
            processed_count += 1
            
            print(f"\n--- Processing entry {processed_count}: {entry.get('id', 'unknown')} ---")
            
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
        }
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
    print(f"\nVisual clues saved to: eval/visualClue/")


if __name__ == "__main__":
    main()