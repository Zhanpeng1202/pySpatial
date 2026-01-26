


task_description = """
    You are now asked to solve a spatial reasoning related problem. The input are image(s) and a natural langugae question that
    specifically designed to test your spatial reasoning ability.
    It is not trivial to solve these tasks directly as a vision langugae model. 
    However, You have access to the following Python API:
"""

api_specification = """
    In the PySpatial API, we explicitly introduce the 3D inductive bias.
    We provide a Scene class that contains the image(s) and a question.
    Further, we also provide a 3D reconstruction process that can be used to generate a 3D point cloud and camera parameters.
    
    class Reconstruction:
        def __init__(self, point_cloud, extrinsics, intrinsics):
            self.point_cloud = point_cloud
            self.extrinsics = extrinsics
            self.intrinsics = intrinsics
        
    class Scene:
        "Simple scene class that holds image data."
        
        def __init__(self, path_to_images: Union[str, List[str]], question: str = ""):
            self.question = question
            self.images = self._load_images(path_to_images)
            self.reconstruction : Reconstruction = None
        
        def _load_images(self, path_to_images: Union[str, List[str]]) -> List[str]:
            "Load image paths from directory or list."
            if isinstance(path_to_images, str):
                if os.path.isdir(path_to_images):
                    # Load all images from directory
                    image_extensions = ['*.png', '*.jpg', '*.jpeg']
                    images = []
                    for ext in image_extensions:
                        images.extend(glob.glob(os.path.join(path_to_images, ext)))
                    return sorted(images)
                else:
                    # Single image file
                    return [path_to_images]
            else:
                # List of image paths
                return list(path_to_images)

    class pySpatial:
        "Simple interface for 3D vision tools."
        # we disable other function for now
        
        @staticmethod
        def reconstruct(scene: Scene):
            "3D reconstruction from scene images."
            
            return reconstruct_3d(scene.images)
        
        @staticmethod
        def describe_camera_motion(recon: Reconstruction):
            "Describe camera motion from reconstruction results.
            Args:
            "
            extrinsics = recon.extrinsics
            return describe_camera_motion(extrinsics)

        @staticmethod
        def synthesize_novel_view(recon: Reconstruction, new_camera_pose):
            "Generate novel view synthesis from reconstruction results.
            Args:
            "
            return novel_view_synthesis(recon)
        
        # methods to manipulate camera pose    
        def rotate_right(extrinsic, angle=np.pi/2):

        def rotate_left(extrinsic, angle=np.pi/2):

        def move_forward(extrinsic, distance=0.1):

        def move_backward(extrinsic, distance=0.1):

        def turn_around(extrinsic):

        
        @staticmethod
        def estimate_depth(image):
            return estimate_depth(image)

    please follow the instructions to generate the code in the ```python ``` block.
    
    ```python
    def program(input_scene: Scene):
        reconstruction3D = pySpatial.reconstruct(input_scene)
        camera_motion = pySpatial.describe_camera_motion(reconstruction3D)
        return camera_motion
    ```

"""

# in-context learning exmaples
example_problems = """    
"""


code_generation_prompt = f"""
    Now please utilize the PySpatial API and write a python function to solve the problem.
    Noted that you can first do reasoning and then write the code. 
    But the code should be wrapped in the ```python ``` block.
    Write a compact code block
    Also, the function written should be named as program and the input parameter should be a Scene object.
    for example,
    ```python
    def program(input_scene: Scene):
        ...
        return ...
    ```
"""


# Prompt template for ReAct: ReAct: Synergizing Reasoning and Acting in Language Models https://arxiv.org/abs/2210.03629

answer_background = f"""
    We are now solving a spatial reasoing problem.     
    It is not trivial to solve these tasks directly as a vision langugae model. 
    However, We have access to the following PySpatial API:
    {api_specification}
    
    We generate a python code based on the PySpatial API to solve this problem.
"""

answer_prompt = """
    Based on the code and the visual clue from the execution, answer the question.
"""




# Prompt for the answer without visual clue
without_visual_clue_background = """
    Solve this spatial reasoning problem based on the question and the image input.
    
    First, analyze the question, extract useful information from the question description, 
    then try to answer it based on the useful visual information.
    
    Give your best guess if you cannot find the best answer.
"""

