


task_description = """
    You are now asked to solve a spatial reasoning related problem. The input are image(s) and a natural langugae question that
    specifically designed to test your spatial reasoning ability.
    It is not trivial to solve these tasks directly as a vision langugae model. 
    However, You have access to the following Python API:
"""

api_specification = """
    class Scene:
        Scene(path_to_images: Union[str, List[str]], question: str="")
            .images: List[str]

    class pySpatial:
        reconstruct(scene: Scene) -> dict
        describe_camera_motion(scene: Scene, reconstruction_result) -> str
"""

# in-context learning exmaples
example_problems = """
    Problem 1:
    Question: "Based on these two views showing the same scene:
    in which direction did I move from the first view to the second view?
    A. Diagonally forward and left
    B. Directly right
    C. Directly left
    D. Diagonally forward and right"
    
    How to solve this problem?
    Step 1: we can easily find the ansewr with camera extrinsics.
    Step 2: therefore, we should first reconstruct the scene, and then use the camera extrinsics to find the answer.
    Step 3: it is still not trivial to directly get the answer from extrinsic matrix.
    Step 4: we can use the pySpatial.describe_camera_motion to get the answer.
    Next, write python code within the pySpatial API provided, then an agent will automatically collect the code I wrote and execute it.
    Step 5: After I get the visual clue from execution, I can easily match the answer:
    
"""