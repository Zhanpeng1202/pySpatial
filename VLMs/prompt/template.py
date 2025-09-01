


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
    
    ```python
    def program(input_scene: Scene):
        reconstruction3D = pySpatial.reconstruct(input_scene)
        camera_motion = pySpatial.describe_camera_motion(reconstruction3D)
        return camera_motion
    ```
    Step 5: After I get the visual clue from execution, I can easily match the answer:
    
    Problem 2:
    Question: "Based on these four images (image 1, 2, 3, and 4) showing the black sneaker from different viewpoints (front, left, back, and right), 
    with each camera aligned with room walls and partially capturing the surroundings: 
    From the viewpoint presented in image 2, what is to the right of the black sneaker?
    A. TV
    B. Wooden dining table 
    C. Light purple sofa
    D. Brown curtains and windows
    
    How to solve this problem?
    We can first reconstruct the scene and get the point cloud.
    After that we can use the pySpatial.synthesize_novel_view to get the novel view. We should specifically design a new camera pose.
    We want to see the right of the black sneaker from the viewpoint as image 2. One possible way is to rotate the camera right from viewpoint 2.  
    We can use the pySpatial.rotate_right to rotate the camera right.
    
    Next, write the python code with the pySptial API provided.
    ```python
    def program(input_scene: Scene):
        reconstruction3D = pySpatial.reconstruct(input_scene)
        novel_viewpoint = pySpatial.rotate_right(reconstruction3D.extrinsics[1]) # noted that the second viewpoint is the 1st index in the array
        novel_view = pySpatial.synthesize_novel_view(reconstruction3D, novel_viewpoint)
        return novel_view
    ```
    From the render view, we can find the result is: 
    
    
    Problem 3:
    Based on these four images (image 1, 2, 3, and 4) showing the pink bottle from different viewpoints (front, left, back, and right),
    with each camera aligned with room walls and partially capturing the surroundings:
    If I am standing at the same spot and facing the same direction as shown in image 1,
    then I turn right and move forward, will I get closer to the pink plush toy and headboard?
    
    since we do not have the way to compare distance in 3D space, we can render two images, and use these two images as visual clue.
    ```python
        reconstructed_scene = pySpatial.reconstruct(input_scene)
        base_viewpoint = reconstructed_scene.extrinsics[0] # the image 1 indicates the 0th index in the array
        
        viewpoint_turn_right = pySpatial.rotate_right(base_viewpoint)
        viewpoint_move_forward = pySpatial.move_forward(viewpoint_turn_right)

        image_right = pySpatial.synthesize_novel_view(reconstructed_scene, viewpoint_turn_right)   
        image_forward = pySpatial.synthesize_novel_view(reconstructed_scene, viewpoint_move_forward)
        
        # we should compare these two images, check if the object exists and if the distance is closer.
        visual_clue = [image_right, image_forward]        
        return visual_clue
    ```
    
    
"""


code_generation_prompt = f"""
    Now please utilize the PySpatial API and write a python function to solve the problem.
    Noted that you can first do reasoning and then write the code. 
    But the code should be wrapped in the ```python ``` block.
    Write a compact code block
    Also, the name of your function written should be program and the input should be a Scene object.
"""


# Prompt template for ReAct: ReAct: Synergizing Reasoning and Acting in Language Models https://arxiv.org/abs/2210.03629

background = f"""
    We are now solving a spatial reasoing problem.     
    It is not trivial to solve these tasks directly as a vision langugae model. 
    However, We have access to the following PySpatial API:
    {api_specification}
    
    We generate a python code based on the PySpatial API to solve this problem.
"""

reflexion = """
    Based on the code and the visual clue from the execution, can we first reason if the visual clue is helpful?
    
    1. Is the code generated correct, which means the code does not use functions outside the PySpatial API?
    2. What is the answer of the question directly based on visual clue? If the answer is contained in the choice, answer it directly.
    3. If the code is not correct or the visual clue is not helpful, we should discard the visual clue and directly answer it with the VLM capabilities.
"""





