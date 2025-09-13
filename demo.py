#!/usr/bin/env python3
"""
Demo script for testing pySpatial Agent with customized single scene data.

This script allows you to:
1. Specify a folder containing images for a single scene
2. Ask a spatial reasoning question about those images
3. Get the complete pipeline results including code generation, execution, and answer

Usage:
    python demo.py --folder /path/to/image/folder --question "Your spatial reasoning question"
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime

# Add parent directory to Python path to import pySpatial_Interface
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pySpatial_Interface import Agent, Scene


def get_image_files(folder_path: str) -> List[str]:
    """
    Get all image files from the specified folder.
    
    Args:
        folder_path: Path to the folder containing images
        
    Returns:
        List of image file paths
    """
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            image_files.append(str(file_path))
    
    if not image_files:
        raise ValueError(f"No image files found in folder: {folder_path}")
    
    # Sort for consistent ordering
    image_files.sort()
    return image_files


def make_json_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, 'save'):
        # PIL Image or similar
        return f"<Image object: {type(obj).__name__}>"
    else:
        return str(obj)


def run_demo(folder_path: str, question: str, agent: Agent, scene_id: Optional[str] = None) -> dict:
    """
    Run the complete pySpatial pipeline on a custom scene.
    
    Args:
        folder_path: Path to folder containing scene images
        question: Spatial reasoning question to ask
        agent: pySpatial Agent instance
        scene_id: Optional scene identifier
        
    Returns:
        Dictionary containing complete pipeline results
    """
    print(f"🔍 Loading images from: {folder_path}")
    images = get_image_files(folder_path)
    print(f"📸 Found {len(images)} images:")
    for i, img in enumerate(images, 1):
        print(f"  {i}. {Path(img).name}")
    
    print(f"\n❓ Question: {question}")
    print("\n" + "="*60)
    
    # Create scene
    if scene_id is None:
        scene_id = f"demo_{Path(folder_path).name}"
    
    scene = Scene(images, question, scene_id=scene_id)
    
    # Step 1: Generate code
    print("🔧 Step 1: Generating code...")
    generated_response = agent.generate_code(scene)
    print(f"✓ Generated response length: {len(generated_response)} characters")
    
    # Parse the response to extract code
    print("\n🔍 Step 2: Parsing code...")
    parsed_code = agent.parse_LLM_response(scene, generated_response)
    parse_success = parsed_code is not None and parsed_code.strip() != ""
    
    if parse_success:
        print("✓ Code parsing successful")
        print(f"📝 Generated code preview:\n{parsed_code[:200]}{'...' if len(parsed_code) > 200 else ''}")
    else:
        print("❌ Code parsing failed")
        return {
            "scene_id": scene_id,
            "folder_path": folder_path,
            "question": question,
            "images": images,
            "generated_response": generated_response,
            "parsed_code": parsed_code,
            "parse_success": False,
            "execution_success": False,
            "answer_generation_success": False,
            "status": "failed_at_parsing"
        }
    
    # Step 3: Execute code to get visual clue
    print("\n⚙️  Step 3: Executing code...")
    visual_clue = agent.execute(scene)
    execution_success = visual_clue != "there is an error during code generation, no visual clue provided"
    
    if execution_success:
        print("✓ Code execution successful")
        print(f"👁️  Visual clue type: {type(visual_clue).__name__}")
    else:
        print("❌ Code execution failed")
        print(f"Error: {visual_clue}")
        return {
            "scene_id": scene_id,
            "folder_path": folder_path,
            "question": question,
            "images": images,
            "generated_response": generated_response,
            "parsed_code": parsed_code,
            "parse_success": True,
            "execution_success": False,
            "execution_error": visual_clue,
            "answer_generation_success": False,
            "status": "failed_at_execution"
        }
    
    # Step 4: Generate answer using visual clue
    print("\n🤔 Step 4: Generating answer...")
    answer_response = agent.answer(scene, visual_clue)
    answer_generation_success = answer_response is not None
    
    if answer_generation_success:
        print("✓ Answer generation successful")
        print(f"💡 Answer: {answer_response.answer}")
        print(f"🧠 Reasoning: {answer_response.reasoning}")
    else:
        print("❌ Answer generation failed")
    
    # Compile results
    result = {
        "scene_id": scene_id,
        "folder_path": folder_path,
        "question": question,
        "images": images,
        "num_images": len(images),
        "generated_response": generated_response,
        "parsed_code": parsed_code,
        "parse_success": parse_success,
        "execution_success": execution_success,
        "visual_clue_type": type(visual_clue).__name__ if visual_clue and execution_success else None,
        "answer_generation_success": answer_generation_success,
        "generated_answer": answer_response.answer if answer_response else None,
        "generated_reasoning": answer_response.reasoning if answer_response else None,
        "status": "success" if answer_generation_success else "partial_success",
        "timestamp": datetime.now().isoformat()
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Demo pySpatial Agent with custom scene data")
    parser.add_argument("--folder", type=str, required=True,
                       help="Path to folder containing scene images")
    parser.add_argument("--question", type=str, required=True,
                       help="Spatial reasoning question to ask about the scene")
    parser.add_argument("--scene_id", type=str, default=None,
                       help="Optional scene identifier (defaults to folder name)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API key (if not provided, uses OPENAI_API_KEY env var)")
    parser.add_argument("--save_result", type=str, default=None,
                       help="Optional file path to save detailed results as JSON")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.folder):
        print(f"❌ Error: Folder does not exist: {args.folder}")
        sys.exit(1)
    
    if not os.path.isdir(args.folder):
        print(f"❌ Error: Path is not a directory: {args.folder}")
        sys.exit(1)
    
    print("🚀 pySpatial Demo - Custom Scene Analysis")
    print("="*50)
    
    # Initialize Agent
    try:
        agent = Agent(api_key=args.api_key)
        print("✓ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        sys.exit(1)
    
    # Run demo
    try:
        result = run_demo(args.folder, args.question, agent, args.scene_id)
        
        print("\n" + "="*60)
        print("📊 FINAL RESULTS")
        print("="*60)
        print(f"Scene ID: {result['scene_id']}")
        print(f"Status: {result['status']}")
        print(f"Images processed: {result['num_images']}")
        print(f"Parse success: {result['parse_success']}")
        print(f"Execution success: {result['execution_success']}")
        print(f"Answer generation success: {result['answer_generation_success']}")
        
        if result.get('generated_answer'):
            print(f"\n🎯 FINAL ANSWER: {result['generated_answer']}")
        
        if result.get('generated_reasoning'):
            print(f"💭 REASONING: {result['generated_reasoning']}")
        
        # Save results if requested
        if args.save_result:
            serializable_result = make_json_serializable(result)
            with open(args.save_result, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            print(f"\n💾 Results saved to: {args.save_result}")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()