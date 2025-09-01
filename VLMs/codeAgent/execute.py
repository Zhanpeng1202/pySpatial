





import re


def parse_LLM_response(response: str):
    """
    Extracts the first python code block (```python ... ```) from text.
    Returns the code as a string, or "" if not found.
    """
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def execute_code(code_str: str):
    """
    Execute a code string and return the defined program function.
    Uses exec approach without safety considerations for simplicity.
    
    Args:
        code_str: String containing Python code that defines a 'program' function
        
    Returns:
        The program function from the executed code
    """
    # Step 1: Convert the escaped newlines (\n) into real newlines
    code_src = code_str.encode().decode("unicode_escape")
    
    # Step 2: Import necessary classes and modules for the execution context
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from pySpatial_Interface import Scene, Reconstruction, pySpatial
    import numpy as np
    
    # Create execution context with necessary imports
    execution_globals = {
        'Scene': Scene,
        'Reconstruction': Reconstruction, 
        'pySpatial': pySpatial,
        'np': np,
        'numpy': np,
        '__builtins__': __builtins__
    }
    
    # Step 3: Execute the code so `program` is defined
    namespace = {}
    exec(code_src, execution_globals, namespace)
    
    # Step 4: Get the function
    program = namespace["program"]
    return program

