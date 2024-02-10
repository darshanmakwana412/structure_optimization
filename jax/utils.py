from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np

def draw_cube(center, side_lengths):

    # Calculate the vertices of the cuboid centered at (1, 1, 1)
    half_sides = np.array(side_lengths) / 2
    vertices = np.array([
        [center[0] - half_sides[0], center[1] - half_sides[1], center[2] - half_sides[2]],
        [center[0] + half_sides[0], center[1] - half_sides[1], center[2] - half_sides[2]],
        [center[0] + half_sides[0], center[1] + half_sides[1], center[2] - half_sides[2]],
        [center[0] - half_sides[0], center[1] + half_sides[1], center[2] - half_sides[2]],
        [center[0] - half_sides[0], center[1] - half_sides[1], center[2] + half_sides[2]],
        [center[0] + half_sides[0], center[1] - half_sides[1], center[2] + half_sides[2]],
        [center[0] + half_sides[0], center[1] + half_sides[1], center[2] + half_sides[2]],
        [center[0] - half_sides[0], center[1] + half_sides[1], center[2] + half_sides[2]]
    ])

    # Define the 6 faces of the cube
    faces = [[vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]], 
            [vertices[0], vertices[3], vertices[7], vertices[4]], 
            [vertices[1], vertices[2], vertices[6], vertices[5]], 
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]]]

    # Create a Poly3DCollection object for the cube
    cube = Poly3DCollection(faces, facecolors='green', linewidths=1, edgecolors='green', alpha=1.0)
    return cube

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)