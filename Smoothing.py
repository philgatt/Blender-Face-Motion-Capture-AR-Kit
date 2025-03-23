import bpy
import numpy as np

class BlendshapeSmoother(bpy.types.Operator):
    """Smooth Blendshape Animations"""
    bl_idname = "object.smooth_blendshapes"
    bl_label = "Smooth Blendshapes"
    
    def execute(self, context):
        factor = context.scene.smooth_factor
        smooth_shape_keys(factor)
        return {'FINISHED'}

class BlendshapeSmoothingPanel(bpy.types.Panel):
    """Creates a Panel in the scene properties"""
    bl_label = "Blendshape Smoothing"
    bl_idname = "SCENE_PT_blendshape_smoothing"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.label(text="Smoothing Factor:")
        row = layout.row()
        row.prop(scene, "smooth_factor", text="Factor")
        
        row = layout.row()
        row.operator("object.smooth_blendshapes")

def smooth_curve(fcurve, factor):
    keyframes = fcurve.keyframe_points
    if len(keyframes) < 3:
        return  # Not enough keyframes to smooth
    
    times = np.array([kp.co[0] for kp in keyframes])
    values = np.array([kp.co[1] for kp in keyframes])
    
    smoothed_values = np.convolve(values, np.ones(factor)/factor, mode='same')
    
    for i, kp in enumerate(keyframes):
        kp.co[1] = smoothed_values[i]
        kp.handle_left_type = 'AUTO'
        kp.handle_right_type = 'AUTO'
    
    fcurve.update()

def smooth_shape_keys(factor):
    obj = bpy.context.active_object
    if not obj or not obj.animation_data:
        print("No active object with animation data found.")
        return
    
    action = obj.animation_data.action
    if not action:
        print("No animation action found.")
        return
    
    for fcurve in action.fcurves:
        if 'key_blocks' in fcurve.data_path:
            smooth_curve(fcurve, factor)
    
    print(f"Smoothed all blendshape animations with factor {factor}.")

def register():
    bpy.utils.register_class(BlendshapeSmoother)
    bpy.utils.register_class(BlendshapeSmoothingPanel)
    bpy.types.Scene.smooth_factor = bpy.props.IntProperty(
        name="Smoothing Factor",
        default=3,
        min=1,
        max=10,
        description="Amount of smoothing applied to blendshapes"
    )

def unregister():
    bpy.utils.unregister_class(BlendshapeSmoother)
    bpy.utils.unregister_class(BlendshapeSmoothingPanel)
    del bpy.types.Scene.smooth_factor

if __name__ == "__main__":
    register()

# Integrate smoothing into existing motion capture pipeline
def apply_blendshape_smoothing():
    factor = bpy.context.scene.smooth_factor
    smooth_shape_keys(factor)

class ApplyBlendshapeSmoothingOperator(bpy.types.Operator):
    """Apply Blendshape Smoothing"""
    bl_idname = "object.apply_blendshape_smoothing"
    bl_label = "Apply Blendshape Smoothing"
    
    def execute(self, context):
        apply_blendshape_smoothing()
        return {'FINISHED'}

bpy.utils.register_class(ApplyBlendshapeSmoothingOperator)
