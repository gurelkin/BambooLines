import pyrender
import trimesh
import open3d
import matplotlib.pyplot as plt
from consts import *


def get_intact_normal(tmesh: trimesh.Trimesh) -> np.array:
    """
    Finds the mean normal of a fragment's intact surface
    :param tmesh: Trimesh instance of the fragment
    :return: the intact surface normal as (3, ) numpy array
    """
    pcloud = open3d.geometry.PointCloud()
    pcloud.points = open3d.utility.Vector3dVector(tmesh.vertices.tolist())
    pcloud.estimate_normals()
    # find the approximated plane and its in-liers
    _, inliers = pcloud.segment_plane(distance_threshold=SEGMENT_PLANE_DIST_THRESH,
                                      ransac_n=SEGMENT_PLANE_RANSAC_N,
                                      num_iterations=SEGMENT_PLANE_NUM_ITER)
    pcloud = pcloud.select_by_index(inliers)
    pcloud.estimate_normals()
    # return the mean of the plane's normals as a unit vector
    normal = np.mean(np.asarray(pcloud.normals), axis=0)
    return normal / np.linalg.norm(normal)


def base_transform(tmesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Transforms a fragment such that its center of mass is on the origin and
    its intact surface is pointing "up" at (0, 0, 1)
    :param tmesh: The fragment's triangle mesh
    :return: The transformed trimesh
    """
    # set the center of mass to be at (0, 0, 0)
    tmesh.apply_translation(-tmesh.vertices.mean(axis=0))
    # rotate the fragment such that the intact surface is pointing "up"
    intact_normal = - get_intact_normal(tmesh)
    rotation_angle = np.arccos(np.dot(intact_normal, [0, 0, 1]))
    rotation_axis = np.cross(intact_normal, [0, 0, 1])
    rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)
    tmesh.apply_transform(rotation_matrix)
    return tmesh


class Render(object):

    def __init__(self, path, pose='intact'):
        self.trimesh = base_transform(trimesh.load_mesh(path))
        self.scene = pyrender.Scene(ambient_light=SCENE_AMBIENT_LIGHT, bg_color=SCENE_BG_COLOR)
        self.scene.add(pyrender.Mesh.from_trimesh(self.trimesh))
        self.set_camera_position(pose)

    def set_camera_position(self, pose='intact'):
        # remove the previous camera and light nodes
        for camera_node in self.scene.camera_nodes:
            self.scene.remove_node(camera_node)
        for light_node in self.scene.directional_light_nodes:
            self.scene.remove_node(light_node)
        # set the camera in front of the fragment's intact / backside
        camera_pose = np.eye(4)
        if pose == 'intact':
            camera_pose[2, 3] = np.max(self.trimesh.vertices[:, 2]) + CAMERA_DISTANCE
        else:
            camera_pose[2, 2] = -1
            camera_pose[2, 3] = np.min(self.trimesh.vertices[:, 2]) - CAMERA_DISTANCE
        # add a camera and light nodes to the scene
        camera = pyrender.PerspectiveCamera(yfov=CAMERA_YFOV)
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        self.scene.add_node(camera_node)
        light = pyrender.DirectionalLight(color=LIGHT_COLOR, intensity=LIGHT_INTENSITY)
        light_node = pyrender.Node(light=light, matrix=camera_pose)
        self.scene.add_node(light_node)

    def plot(self):
        """
        Render an image taken from the camera in the render's scene
        """
        color, _ = pyrender.OffscreenRenderer(viewport_width=VP_WIDTH, viewport_height=VP_HEIGHT).render(self.scene)
        plt.imshow(color)
        plt.show()

    def view(self):
        """
        Render an interactive window where the camera and the light are positioned as set
        """
        pyrender.Viewer(self.scene).render_lock.release()



