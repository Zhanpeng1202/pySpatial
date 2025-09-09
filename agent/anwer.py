import os
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
from agent.prompt.template import answer_background, answer_prompt, without_visual_clue_background
from pySpatial_Interface import Scene
import numpy as np
from pydantic import BaseModel
from typing import Literal


def o3d_image_to_data_url(o3d_img: "o3d.geometry.Image") -> str:
    """Convert Open3D Image -> PNG data URL suitable for OpenAI vision input."""
    arr = np.asarray(o3d_img)  # Open3D Image to NumPy, noted that the data type is int8
    
    # Handle different image formats
    if len(arr.shape) == 3:  # Color image
        if arr.shape[2] == 3:
            pil = Image.fromarray(arr, mode="RGB")
        elif arr.shape[2] == 4:
            pil = Image.fromarray(arr, mode="RGBA")
        else:
            raise ValueError(f"Unexpected number of channels {arr.shape[2]}")
    elif len(arr.shape) == 2:  # Grayscale image
        pil = Image.fromarray(arr, mode="L")
    else:
        raise ValueError(f"Unexpected image shape {arr.shape}")
    
    buf = BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class SpatialAnswer(BaseModel):
    reasoning: str
    answer: Literal["A", "B", "C", "D"]


def answer(scene: Scene, api_key: str = None):
    """
    Generate structured answer using OpenAI API model based on the scene question.
    
    Args:
        scene: Scene object containing the question
        api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
    
    Returns:
        dict: Structured response with 'reasoning' and 'answer' fields.
    """
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    client = OpenAI(api_key=api_key)
    
    base_prompt = f"""
        {answer_background}
        {answer_prompt}
    """
        
    query_for_vlm = f"""
        {base_prompt}
        the question is {scene.question}
        the generated code is {scene.code}
        the visual clue is pasted below:
        
    """
    
    visual_clue = scene.visual_clue
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes visual information and answers spatial reasoning questions based on the generated visual clue."}
    ]
    
    # Handle visual clues based on type
    if visual_clue is None:
        # No visual clue, just add the text query
        messages.append({"role": "user", "content": query_for_vlm})
    elif isinstance(visual_clue, str):
        # Case 1: Visual clue is a string
        combined_query = f"{query_for_vlm}\n\nVisual clue: {visual_clue}"
        messages.append({"role": "user", "content": combined_query})
    elif isinstance(visual_clue, list):
        # Case 2: Visual clue is list of Open3D images
        messages.append({"role": "user", "content": query_for_vlm})
        for o3d_img in visual_clue:
            print(f"-----Type of list item: {type(o3d_img)}")
            data_url = o3d_image_to_data_url(o3d_img)
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": data_url,
                    }
                ]
            })
    else:
        # Case 3: Visual clue is a single Open3D image
        messages.append({"role": "user", "content": query_for_vlm})
        data_url = o3d_image_to_data_url(visual_clue)
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": data_url,
                }
            ]
        })
    
    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=messages,
        max_output_tokens=2000,
        text_format=SpatialAnswer
    )
    
    print(f"--------------------------------{response.output_parsed}")
    return response.output_parsed


def answer_without_visual_clue(scene: Scene, api_key: str = None):
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    client = OpenAI(api_key=api_key)
    
    
    base_prompt = f"""
        {without_visual_clue_background}
        the question is {scene.question}
    """
    
    # get the images from scene.images
