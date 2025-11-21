import os
from openai import OpenAI
from agent.prompt.template import task_description, api_specification, example_problems, code_generation_prompt
from pySpatial_Interface import Scene


# TODO: Rewrite the codeAgent with structured output with pydantic


def generate_code_from_query(scene: Scene, api_key: str = None):
    """
    Generate code using OpenAI GPT-4 model based on the scene question.
    
    Args:
        scene: Scene object containing the question
        api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
    
    Returns:
        str: Generated code response from GPT-4
    """
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    client = OpenAI(api_key=api_key)
    
    base_prompt = f"""
        {task_description}
        {api_specification}
        {example_problems}
    """
        
    query_for_vlm = f"""
        {base_prompt}
        {code_generation_prompt}
        the question is {scene.question}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates Python code using the pySpatial API to solve spatial reasoning problems."},
                {"role": "user", "content": query_for_vlm}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        raise Exception(f"Error calling OpenAI API: {str(e)}")

def generate_code(scene: Scene, api_key: str = None):
    """Legacy function name for backward compatibility"""
    return generate_code_from_query(scene, api_key)


