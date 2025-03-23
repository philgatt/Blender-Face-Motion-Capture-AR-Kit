import bpy
import sys
import pip
import msvc_runtime

# model = D:\Blender\MotionCapture\face_landmarker.task

def main(context, preview, head_rotation):
    
    model = context.scene.model_path

    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import keyboard
    from mathutils import Vector, Euler

    def draw_landmarks_on_image(rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

        return annotated_image

    def plot_face_blendshapes_bar_graph(face_blendshapes):
        face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
        face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
        face_blendshapes_ranks = range(len(face_blendshapes_names))

        fig, ax = plt.subplots(figsize=(12, 12))
        bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
        ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
        ax.invert_yaxis()

        for score, patch in zip(face_blendshapes_scores, bar.patches):
            plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

        ax.set_xlabel('Score')
        ax.set_title("Face Blendshapes")
        plt.tight_layout()
        plt.show()

    import bpy

    def set_blendshape_value(mesh, name, value, 
frame):
        for mesh in mesh:
            shape_key = mesh.shape_keys.key_blocks.get(name)
            if shape_key is not None:
                shape_key.value = value
                shape_key.keyframe_insert(data_path="value", frame=frame)

    mesh = [bpy.data.objects["head"].data,
            bpy.data.objects["eyeLeft"].data,
            bpy.data.objects["eyeRight"].data]

    # STEP 2: Create an FaceLandmarker object.
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True, output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # 
    # STEP 1: Import the necessary modules.
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mathutils import Matrix

    # STEP 2: Create an FaceLandmarker object.
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # Define a function to process each frame.
    def process_frame(frame, frame_num):  # Changed i to frame_num
        # STEP 3: Load the input image.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # STEP 4: Detect face landmarks from the input image.
        detection_result = detector.detect(image)
        trans_matrix = detection_result.facial_transformation_matrixes

        obj = bpy.data.objects["head"]
        bpy.context.scene.frame_set(frame_num)  # Set the Blender frame

        if preview == True:
            # STEP 5: Process the detection result.
            # In this case, visualize it.
            annotated_image = draw_landmarks_on_image(frame, detection_result)
            cv2.imshow("Facial Landmarks", annotated_image)

        # Apply the blendshape values to the 3D character.
        if detection_result.face_blendshapes:
            if len(detection_result.face_blendshapes) > 0:
                blendshapes = detection_result.face_blendshapes[0]
                for bs in blendshapes:
                    set_blendshape_value(mesh, bs.category_name, bs.score, frame_num)  # Use frame_num
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            print("blendshapes set")

        if head_rotation == True:
            if detection_result.facial_transformation_matrixes:
                trans_matrix = detection_result.facial_transformation_matrixes[0]
                matrix = Matrix(trans_matrix)
                current_rotation = matrix.to_euler()

                # Apply smoothing
                smoothed_rotation = current_rotation.copy()
                if context.scene.head_rotation_smoothing > 0.0:
                    previous_rotations.append(current_rotation)
                    if len(previous_rotations) > 10:  # You can adjust the number of frames to average
                        previous_rotations.pop(0)

                    # Calculate weighted average for Euler angles
                    total_weight = sum(range(1, len(previous_rotations) + 1))
                    smoothed_rotation = Euler((0, 0, 0), 'XYZ')  # Initialize with zero rotation

                    for i, rot in enumerate(previous_rotations):
                        weight = (i + 1) / total_weight  # Linear increasing weight
                        smoothed_rotation.x += rot.x * weight
                        smoothed_rotation.y += rot.y * weight
                        smoothed_rotation.z += rot.z * weight
                else:
                    smoothed_rotation = current_rotation  # If no smoothing, use current rotation

                obj.rotation_euler = smoothed_rotation
                obj.keyframe_insert(data_path="rotation_euler", frame=frame_num)  # Keyframe rotation

    # Define the video capture device (0 for the default camera).
    #cap = cv2.VideoCapture(0)
    if context.scene.use_video_file:
        cap = cv2.VideoCapture(context.scene.input_video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        cap = cv2.VideoCapture(0)
        frame_count = 1000 #default value

    start_frame = bpy.context.scene.frame_start
    end_frame = bpy.context.scene.frame_end

    bpy.context.scene.frame_end = frame_count

    frame_num = start_frame
    previous_rotations = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error")
            break
        process_frame(frame, frame_num)
        frame_num += 1

        if keyboard.is_pressed('q'):
            print("Q was pressed")
            break

        # Check for the 'q' key to 
        #quit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close the windows.
    cap.release()
    cv2.destroyAllWindows()
    return frame_num  # Return the last frame number

class SimpleOperator(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.simple_operator"
    bl_label = "Track Face"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        preview = context.scene.preview
        head_rotation = context.scene.head_rotation
        last_frame = main(context, preview, head_rotation) #get the last frame number
        bpy.context.scene.frame_end = last_frame
        bpy.ops.screen.animation_play() #play the animation
        return {'FINISHED'}

def menu_func(self, context):
    self.layout.operator(SimpleOperator.bl_idname, text=SimpleOperator.bl_label)

class LayoutDemoPanel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "Layout Demo"
    bl_idname = "SCENE_PT_layout"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.label(text="Input:")
        row = layout.row()
        row.prop(scene, "use_video_file", text="Use Video File")

        if scene.use_video_file:
            row = layout.row()
            row.prop(scene, "input_video_path", text="")

        # New row for 'model path'
        layout.label(text="Landmarks Model Path:")
        row = layout.row()
        row.prop(scene, "model_path", text="")

        # Big render button
        layout.label(text="Face Capture:")
        row = layout.row()
        row.scale_y = 2.0
        row.operator("object.simple_operator")

        layout.label(text="Preview:")
        row = layout.row()
        row.prop(scene, "preview", text="Enable Preview")

        layout.label(text="Head rotation:")
        row = layout.row()
        row.prop(scene, "head_rotation", text="Enable Head Rotation")
        
        layout.label(text="Head Rotation Smoothing:")  # Added smoothing control
        row = layout.row()
        row.prop(scene, "head_rotation_smoothing", text="Smoothing")


def register():
    bpy.types.Scene.preview = bpy.props.BoolProperty(
        name="Enable Preview",
        default=False)
    bpy.types.Scene.head_rotation = bpy.props.BoolProperty(
        name="Head rotation",
        default=False)
    bpy.types.Scene.input_video_path = bpy.props.StringProperty(
        name="Input Video",
        default="",
        subtype='FILE_PATH')
    bpy.types.Scene.model_path = bpy.props.StringProperty(
            name="Model Path",
            description="Path to the mediapipe landmarks model",
            subtype='FILE_PATH'  # This ensures it's a file path input
    )
    bpy.types.Scene.use_video_file = bpy.props.BoolProperty(
        name="Use Video File",
        default=False
    )
    bpy.types.Scene.head_rotation_smoothing = bpy.props.FloatProperty( #register the smoothing property
        name="Head Rotation Smoothing",
        default=0.5,
        min=0.0,
        max=1.0,
        description="Amount of smoothing applied to head rotation"
    )
    bpy.utils.register_class(SimpleOperator)
    bpy.types.VIEW3D_MT_object.append(menu_func)
    bpy.utils.register_class(LayoutDemoPanel)

def unregister():
    bpy.utils.unregister_class(SimpleOperator)
    bpy.types.VIEW3D_MT_object.remove(menu_func)
    bpy.utils.unregister_class(LayoutDemoPanel)
    del bpy.types.Scene.preview
    del bpy.types.Scene.head_rotation
    del bpy.types.Scene.input_video_path
    del bpy.types.Scene.use_video_file
    del bpy.types.Scene.head_rotation_smoothing # Unregister the smoothing property

if __name__ == "__main__":
    register()