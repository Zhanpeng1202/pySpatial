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
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add parent directory to Python path to import pySpatial_Interface
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pySpatial_Interface import Agent, Scene


def process_scene_with_agent(entry: Dict[str, Any], agent: Agent) -> Dict[str, Any]:
    """
    Process a single JSONL entry as a scene and generate code using the Agent.
    
    Args:
        entry: JSONL entry containing scene information
        agent: pySpatial Agent instance
        
    Returns:
        Dictionary containing the results
    """
    try:
        scene_id = entry['id']
        question = entry.get('question', '')
        images = entry.get('images', [])
        
        # Create Scene object (using dummy images since we don't have actual image paths)
        scene = Scene(images if images else [], question)
        
        # Generate code using the agent
        generated_response = agent.generate_code(scene)
        
        # Parse the response to extract code patterns
        parsed_code = agent.parse_LLM_response(generated_response)
        parse_success = parsed_code is not None and parsed_code.strip() != ""
        
        result = {
            "scene_id": scene_id,
            "question": question,
            "generated_response": generated_response,
            "parsed_code": parsed_code,
            "parse_success": parse_success,
            "status": "success"
        }
        
        print(f"✓ Scene {scene_id}: Successfully generated code")
        return result
        
    except Exception as e:
        result = {
            "scene_id": entry.get('id', 'unknown'),
            "question": entry.get('question', ''),
            "generated_response": None,
            "parsed_code": None,
            "parse_success": False,
            "status": "error",
            "error_message": str(e)
        }
        
        print(f"✗ Scene {entry.get('id', 'unknown')}: Error - {str(e)}")
        return result


def main():
    parser = argparse.ArgumentParser(description="Test pySpatial Agent code generation on MindCube scenes")
    parser.add_argument("--jsonl_path", type=str, 
                       default="/data/Datasets/MindCube/data/raw/MindCube_tinybench.jsonl",
                       help="Path to JSONL file containing scene information")
    parser.add_argument("--output_file", type=str,
                       default="agent_code_generation_results_v3.json",
                       help="Output file path for results")
    parser.add_argument("--max_entries", type=int, default=20,
                       help="Maximum number of entries to process (for testing)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API key (if not provided, uses OPENAI_API_KEY env var)")
    parser.add_argument("--translation_only", action="store_true", default=False,
                       help="Only process translation questions (default: process all questions)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl_path):
        raise ValueError(f"JSONL file not found: {args.jsonl_path}")
    
    # Initialize Agent
    try:
        agent = Agent(api_key=args.api_key)
        print("✓ Agent initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Agent: {str(e)}")
        return
    
    print(f"Processing JSONL file: {args.jsonl_path}")
    print(f"Output file: {args.output_file}")
    print(f"Max entries: {args.max_entries or 'unlimited'}")
    print("="*60)
    
    # Process JSONL entries
    processed_count = 0
    successful_count = 0
    failed_count = 0
    results = []
    
    with open(args.jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if args.max_entries and processed_count >= args.max_entries:
                print(f"Reached maximum entries limit: {args.max_entries}")
                break
            
            try:
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
                
                if result["status"] == "success":
                    successful_count += 1
                else:
                    failed_count += 1
                    
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON on line {line_num}")
                failed_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {str(e)}")
                failed_count += 1
    
    # Save results to current directory
    output_path = Path.cwd() / args.output_file
    
    summary = {
        "total_processed": processed_count,
        "successful": successful_count,
        "failed": failed_count,
        "processing_timestamp": datetime.now().isoformat(),
        "jsonl_source": args.jsonl_path,
        "results": results
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    except Exception as e:
        print(f"\n✗ Failed to save results: {str(e)}")
    
    print(f"\n=== Agent Code Generation Complete ===")
    print(f"Total entries processed: {processed_count}")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {successful_count/processed_count*100:.1f}%" if processed_count > 0 else "N/A")


if __name__ == "__main__":
    main()