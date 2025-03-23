import bpy
from . import MainFaceMocap
from . import Smoothing

bl_info = {
    "name": "Face Motion Capture",
    "author": "Philipp",
    "version": (1, 0, 0),
    "blender": (3, 3, 0),  # Minimum Blender version
    "location": "View3D > Add > Motion Capture",
    "description": "Captures facial motion and applies it to a Blender character.",
    "warning": "",
    "doc_url": "",
    "category": "Motion Capture",
}

def register():
    MainFaceMocap.register()
    Smoothing.register()

def unregister():
    MainFaceMocap.unregister()
    Smoothing.unregister()

if __name__ == "__main__":
    register()