


task_description = """
    You are asked to solve a spatial reasoning problem. The input is one or more images plus a
    natural language question designed to test spatial reasoning, which is hard to answer directly
    as a vision language model. To help, you have access to the following Python API:
"""

api_specification = """
    This is the pySpatial API. It provides 3D reconstruction tools to help solve the problem.
    A Scene holds the input image(s) and the question.

    class Reconstruction:
        # point_cloud : numpy array of 3D points
        # extrinsics  : array/list of per-view camera matrices, shape (N, 4, 4) or (N, 3, 4);
        #               extrinsics[i] is the camera pose of view i (0-based: image 1 -> index 0)
        # intrinsics  : camera intrinsics, may be None
        def __init__(self, point_cloud, extrinsics, intrinsics):
            self.point_cloud = point_cloud
            self.extrinsics = extrinsics
            self.intrinsics = intrinsics

    class Scene:
        "Holds the input image(s) and the question."
        def __init__(self, path_to_images: Union[str, List[str]], question: str = ""):
            self.question = question
            self.images = self._load_images(path_to_images)
            self.reconstruction : Reconstruction = None

    class pySpatial:
        "Interface for 3D vision tools."

        @staticmethod
        def reconstruct(scene: Scene) -> Reconstruction:
            "Reconstruct the scene from its images. Returns a Reconstruction (point_cloud, extrinsics, intrinsics)."

        @staticmethod
        def describe_camera_motion(recon: Reconstruction) -> str:
            "Analyze the camera trajectory across views and return a text description of the camera motion."

        @staticmethod
        def synthesize_novel_view(recon: Reconstruction, new_camera_pose, width=512, height=512, out_path=None):
            "Render the scene from a new viewpoint (a 3x4 or 4x4 extrinsic matrix).
            Returns an image object when out_path is omitted, or a path string when out_path is given.
            For visual clues, do NOT pass out_path so an image object is returned."

        # Camera-pose helpers. rotate_right/rotate_left/turn_around take an optional recon argument
        # used to compute the rotation axis from all views (recommended). When angle/distance are
        # omitted, a sensible default step is used.
        def rotate_right(extrinsic, angle=None, recon=None):

        def rotate_left(extrinsic, angle=None, recon=None):

        def move_forward(extrinsic, distance=None):

        def move_backward(extrinsic, distance=None):

        def turn_around(extrinsic, recon=None):


    Follow the instructions and generate the code inside a ```python ``` block, for example:

    ```python
    def program(input_scene: Scene):
        reconstruction3D = pySpatial.reconstruct(input_scene)
        camera_motion = pySpatial.describe_camera_motion(reconstruction3D)
        return camera_motion
    ```
"""

# in-context learning examples (indexing is 0-based: image 1 -> extrinsics[0])
example_problems = """
    Problem 1:
    Question: "Based on these two views showing the same scene:
    in which direction did I move from the first view to the second view?
    A. Diagonally forward and left
    B. Directly right
    C. Directly left
    D. Diagonally forward and right"

    How to solve: the answer follows from the camera extrinsics, so reconstruct the scene first.
    Reading direction directly off the extrinsic matrices is hard, so use
    pySpatial.describe_camera_motion to get a text description of the motion.

    ```python
    def program(input_scene: Scene):
        reconstruction3D = pySpatial.reconstruct(input_scene)
        camera_motion = pySpatial.describe_camera_motion(reconstruction3D)
        return camera_motion
    ```
    The returned motion description is the visual clue; match it against the options to choose the answer.

    Problem 2:
    Question: "Based on these four images (image 1, 2, 3, and 4) showing the black sneaker from different viewpoints (front, left, back, and right),
    with each camera aligned with room walls and partially capturing the surroundings:
    From the viewpoint presented in image 2, what is to the right of the black sneaker?
    A. TV
    B. Wooden dining table
    C. Light purple sofa
    D. Brown curtains and windows"

    How to solve: reconstruct the scene, then render what is to the right of viewpoint 2 by rotating
    that camera to the right with pySpatial.rotate_right and synthesizing the novel view.

    ```python
    def program(input_scene: Scene):
        reconstruction3D = pySpatial.reconstruct(input_scene)
        novel_viewpoint = pySpatial.rotate_right(reconstruction3D.extrinsics[1])  # image 2 -> index 1
        novel_view = pySpatial.synthesize_novel_view(reconstruction3D, novel_viewpoint)
        return novel_view
    ```
    The rendered image is the visual clue; inspect it to identify what lies to the right of the sneaker.

    Problem 3:
    Question: "Based on these four images (image 1, 2, 3, and 4) showing the pink bottle from different viewpoints (front, left, back, and right),
    with each camera aligned with room walls and partially capturing the surroundings:
    If I am standing at the same spot and facing the same direction as shown in image 1,
    then I turn right and move forward, will I get closer to the pink plush toy and headboard?"

    How to solve: there is no direct API to compare distances in 3D, so render two views (after turning
    right, and after also moving forward) and return both as the visual clue for comparison.

    ```python
    def program(input_scene: Scene):
        reconstructed_scene = pySpatial.reconstruct(input_scene)
        base_viewpoint = reconstructed_scene.extrinsics[0]  # image 1 -> index 0

        viewpoint_turn_right = pySpatial.rotate_right(base_viewpoint)
        viewpoint_move_forward = pySpatial.move_forward(viewpoint_turn_right)

        image_right = pySpatial.synthesize_novel_view(reconstructed_scene, viewpoint_turn_right)
        image_forward = pySpatial.synthesize_novel_view(reconstructed_scene, viewpoint_move_forward)

        # compare these two images: check whether the target objects appear and look closer.
        visual_clue = [image_right, image_forward]
        return visual_clue
    ```
"""


code_generation_prompt = f"""
    Now use the pySpatial API to write a Python function that solves the problem.
    You may reason first, but the final code must be a single compact ```python ``` block.
    The function must be named program and take one Scene argument:
    ```python
    def program(input_scene: Scene):
        ...
        return ...
    ```
    When no existing API directly answers the question, render one or two images that best help solve it,
    and return them (a single image or a list) as the visual clue.
"""


# Prompt template for ReAct: ReAct: Synergizing Reasoning and Acting in Language Models https://arxiv.org/abs/2210.03629

answer_background = f"""
    We are solving a spatial reasoning problem that is hard to answer directly as a vision language model.
    To help, we have the following pySpatial API:
    {api_specification}

    A Python program was generated with this API and executed to produce a visual clue.
"""

answer_prompt = """
    Using the generated code and the visual clue from its execution, answer the question.
    Ground your reasoning in the visual clue and the given options, then return your reasoning
    and the single best option letter (A, B, C, or D).
"""




# Prompt for the answer without visual clue
without_visual_clue_background = """
    Solve this spatial reasoning problem from the question and the input image(s).

    First analyze the question and extract the useful information, then answer based on the
    relevant visual information. Give your best guess if you cannot determine the answer.
"""

