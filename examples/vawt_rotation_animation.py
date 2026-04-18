import bpy
import math

# Configuration
OBJECT_NAME = "final vawt v1"  # Change this if your object has a different name
DURATION_SECONDS = 10
FPS = 30
ROTATION_AXIS = 'Z'  # Vertical axis
DIRECTION = -1  # Counterclockwise (negative rotation on Z-axis when viewed from above)

# Calculate total frames
total_frames = DURATION_SECONDS * FPS

# Set scene frame rate
bpy.context.scene.render.fps = FPS

# Set the frame range
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = total_frames

# Try to find the object by name, otherwise use selected object
if OBJECT_NAME in bpy.data.objects:
    obj = bpy.data.objects[OBJECT_NAME]
    # Make sure it's selected and active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
else:
    # Use currently selected object
    obj = bpy.context.active_object
    if obj is None:
        print("ERROR: No object found. Please select an object or check the object name.")
    else:
        print(f"Object '{OBJECT_NAME}' not found. Using selected object: '{obj.name}'")

if obj:
    # Clear any existing rotation animation on the Z-axis
    if obj.animation_data and obj.animation_data.action:
        for fcurve in obj.animation_data.action.fcurves:
            if fcurve.data_path == 'rotation_euler' and fcurve.array_index == 2:
                obj.animation_data.action.fcurves.remove(fcurve)
    
    # Store the initial rotation
    initial_rotation_z = obj.rotation_euler.z
    
    # Set keyframe at the start (frame 1)
    bpy.context.scene.frame_set(1)
    obj.rotation_euler.z = initial_rotation_z
    obj.keyframe_insert(data_path="rotation_euler", index=2, frame=1)
    
    # Set keyframe at the end (full 360-degree rotation)
    bpy.context.scene.frame_set(total_frames)
    obj.rotation_euler.z = initial_rotation_z + (DIRECTION * 2 * math.pi)
    obj.keyframe_insert(data_path="rotation_euler", index=2, frame=total_frames)
    
    # Set interpolation to linear for smooth constant rotation
    if obj.animation_data and obj.animation_data.action:
        for fcurve in obj.animation_data.action.fcurves:
            if fcurve.data_path == 'rotation_euler' and fcurve.array_index == 2:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = 'LINEAR'
    
    # Reset to frame 1
    bpy.context.scene.frame_set(1)
    
    print(f"✓ Animation created successfully!")
    print(f"  Object: {obj.name}")
    print(f"  Duration: {DURATION_SECONDS} seconds ({total_frames} frames)")
    print(f"  FPS: {FPS}")
    print(f"  Direction: {'Counterclockwise' if DIRECTION < 0 else 'Clockwise'}")
    print(f"  Press SPACE or use Timeline to play the animation")
else:
    print("ERROR: Could not create animation - no valid object found")
