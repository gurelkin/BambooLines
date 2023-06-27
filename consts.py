import numpy as np

# File extensions
OBJ = ".obj"
PLY = ".ply"
JSON = ".json"

# RANSAC parameters for PointCloud.segment_plane()
SEGMENT_PLANE_DIST_THRESH = 0.75
SEGMENT_PLANE_RANSAC_N = 3
SEGMENT_PLANE_NUM_ITER = 10**4


# Rendering parameters
SCENE_AMBIENT_LIGHT = [128, 128, 128]
SCENE_BG_COLOR = [0, 0, 0]
CAMERA_YFOV = np.pi/2.0
CAMERA_DISTANCE = 64
LIGHT_COLOR = [255, 255, 255]
LIGHT_INTENSITY = 4
VP_WIDTH = 800
VP_HEIGHT = 600
