#!/usr/bin/env python3
"""
Parse generated code from agent_code_generation_results_v2.json using Agent.parse_LLM_response()
and save the extracted code to parsed_output.json
"""

import json
import os
from pathlib import Path
from pySpatial_Interface import Agent

def parse_generated_code(input_file="agent_code_generation_results_v2.json", output_file="parsed_output.json"):
    """
    Parse generated code from the results file and save extracted code to output file.
    
    Args:
        input_file: Path to the input JSON file with generated code
        output_file: Path to the output JSON file for parsed results
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Initialize Agent to use parse_LLM_response method
    agent = Agent()
    
    # Load the input JSON file
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract and parse code from each result
    parsed_results = []
    total_entries = len(data.get('results', []))
    successful_parse = 0
    failed_parse = 0
    
    print(f"Processing {total_entries} entries...")
    
    for i, result in enumerate(data.get('results', []), 1):
        scene_id = result.get('scene_id', 'unknown')
        question = result.get('question', '')
        generated_code = result.get('generated_code', '')
        status = result.get('status', 'unknown')
        
        print(f"Processing {i}/{total_entries}: {scene_id}")
        
        # Parse the generated code using Agent's method
        if generated_code and status == 'success':
            try:
                parsed_code = agent.parse_LLM_response(generated_code)
                
                parsed_result = {
                    "scene_id": scene_id,
                    "question": question,
                    "raw_generated_code": generated_code,
                    "parsed_code": parsed_code,
                    "parse_status": "success" if parsed_code else "no_code_found",
                    "original_status": status
                }
                
                if parsed_code:
                    successful_parse += 1
                    print(f"  ✓ Successfully extracted code ({len(parsed_code)} chars)")
                else:
                    failed_parse += 1
                    print(f"  ⚠ No Python code block found")
                    
            except Exception as e:
                parsed_result = {
                    "scene_id": scene_id,
                    "question": question,
                    "raw_generated_code": generated_code,
                    "parsed_code": "",
                    "parse_status": "error",
                    "parse_error": str(e),
                    "original_status": status
                }
                failed_parse += 1
                print(f"  ✗ Parse error: {str(e)}")
        else:
            # Handle failed generation cases
            parsed_result = {
                "scene_id": scene_id,
                "question": question,
                "raw_generated_code": generated_code,
                "parsed_code": "",
                "parse_status": "generation_failed",
                "original_status": status
            }
            if status != 'success':
                parsed_result["error_message"] = result.get('error_message', '')
            failed_parse += 1
            print(f"  - Skipped (original status: {status})")
        
        parsed_results.append(parsed_result)
    
    # Create output summary
    output_data = {
        "summary": {
            "total_entries": total_entries,
            "successful_parse": successful_parse,
            "failed_parse": failed_parse,
            "parse_success_rate": f"{successful_parse/total_entries*100:.1f}%" if total_entries > 0 else "0.0%",
            "input_file": input_file,
            "output_file": output_file
        },
        "parsed_results": parsed_results
    }
    
    # Save to output file
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print(f"\n=== Parsing Complete ===")
    print(f"Total entries: {total_entries}")
    print(f"Successfully parsed: {successful_parse}")
    print(f"Failed to parse: {failed_parse}")
    print(f"Success rate: {successful_parse/total_entries*100:.1f}%" if total_entries > 0 else "0.0%")
    print(f"Output saved to: {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse generated code from agent results")
    parser.add_argument("--input_file", type=str, 
                       default="agent_code_generation_results_v2.json",
                       help="Input JSON file with generated code")
    parser.add_argument("--output_file", type=str,
                       default="parsed_output.json", 
                       help="Output JSON file for parsed results")
    
    args = parser.parse_args()
    
    try:
        parse_generated_code(args.input_file, args.output_file)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()