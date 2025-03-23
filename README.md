# Blender Face Motion Capture AR Kit

This project enables real-time face motion capture in Blender using the **Mediapipe** library and **ARKit shape keys**. The addon integrates face tracking and motion capture into Blender, making it easier to animate facial expressions on 3D models with ARKit blendshapes.

https://github.com/user-attachments/assets/952328c8-7bb1-40f7-91d8-1f0ae98d7b34
## Features
- **Real-time Face Motion Capture**: Uses the **Mediapipe** library for accurate face tracking and expression capture.
- **ARKit Shape Keys**: Motion capture data can be applied to a 3D model with ARKit shape keys, allowing for realistic facial animation.
- **Blendshape Smoothing**: Included feature for smoothing blendshape transitions to improve animation quality.
- **Live Stream or Pre-recorded Video**: Capture facial motion using a webcam in real-time or from a pre-recorded video file.

## Prerequisites

Before using this addon, you need to ensure that the **Mediapipe** library is installed in Blender's Python executable. 

### Installation Instructions
1. Follow this [tutorial video](https://www.youtube.com/watch?v=k1gCIezKA8E) to install **Mediapipe** into Blender's Python environment.
2. Download a 3D mesh with ARKit shape keys, or use the provided example model. You can get it [here](https://github.com/user-attachments/assets/3998f09d-0c24-4bbf-9dcd-e7f22dff8724).

## How to Use

1. **Install the Addon**: 
   - Download and install the addon into Blender.
   - Enable the addon from the **'Scene'** tab on the side panel.
   
   ![Side Panel](https://github.com/user-attachments/assets/b26c4a61-abc6-49fa-a266-07f39ca0b938)
   
2. **Capture Facial Motion**:
   - Choose whether to use a **webcam** for live motion capture or a **pre-recorded video**.
   - The addon will track facial movements and apply them to the selected 3D model with ARKit blendshapes.
   
3. **Blendshape Smoothing**:
   - For smoother transitions between shapes, enable the **Blendshape smoothing feature** for better results.


## Additional Notes

- **Mediapipe Installation**: Make sure that **Mediapipe** is installed correctly in Blender's Python environment before proceeding. This can be done via the methods shown in the linked tutorial.
- **Blendshape Models**: The addon works with any 3D model that has ARKit-compatible shape keys. Download the example model to test and modify.

![Example Animation](https://github.com/user-attachments/assets/46b6eaae-f53a-48bb-bc4f-42cc81737b25)



