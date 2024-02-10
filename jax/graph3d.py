from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt
from utils import draw_cube, Arrow3D
import jax.numpy as np
from jax import grad, config
import imageio
import numpy
import os

# config.update("jax_debug_nans", True)

class Graph3d:
    def __init__(self, nodes, edges):
        self.N = len(nodes)
        self.M = len(edges)

        self.nodes = nodes
        self.edges = edges
        self.loads = []
        self.anchors = []

        self.mask = numpy.ones_like(self.nodes)

    def add_load(self, idx: int, fx: float = 0, fy: float = 0, fz: float = 0):
        self.loads.append((idx, fx, fy, fz))
        self.mask[idx, :] = [0, 0, 0]

    def add_anchor(self, idx: int, sx: bool = True, sy: bool = True, sz: bool = True):
        self.anchors.append((idx, sx, sy, sz))
        self.mask[idx, :] = [not sx, not sy, not sz]

    def proj(self, dx, dy, dz):
        hypot = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return dx / hypot, dy / hypot, dz / hypot

    def calculate_forces(self, nodes):

        A = np.zeros((3 * self.N, self.M))
        for idx, (n1, n2) in enumerate(self.edges):
            n1_x, n1_y, n1_z = nodes[n1]
            n2_x, n2_y, n2_z = nodes[n2]
            ux, uy, uz = self.proj(n1_x - n2_x, n1_y - n2_y, n1_z - n2_z)

            # # towards N1
            # A[n1 * 2][idx], A[n1 * 2 + 1][idx] = ux, uy

            # # towards N2
            # A[n2 * 2][idx], A[n2 * 2 + 1][idx] = -ux, -uy

            A = A.at[n1 * 3, idx].set(ux)
            A = A.at[n1 * 3 + 1, idx].set(uy)
            A = A.at[n1 * 3 + 2, idx].set(uz)
            A = A.at[n2 * 3, idx].set(-ux)
            A = A.at[n2 * 3 + 1, idx].set(-uy)
            A = A.at[n2 * 3 + 2, idx].set(-uz)


        L = np.zeros((3 * self.N, 1))
        for idx, fx, fy, fz in self.loads:
            # L[idx * 2] += fx
            # L[idx * 2 + 1] += fy

            L = L.at[idx * 3].add(fx)
            L = L.at[idx * 3 + 1].add(fy)
            L = L.at[idx * 3 + 2].add(fz)

        for idx, sx, sy, sz in self.anchors:
            if sx:
                # A[idx * 2, :] = 0.0
                # L[idx * 2] = 0.0
                A = A.at[idx * 3, :].set(0.0)
                L = L.at[idx * 3].add(0.0)
            if sy:
                # A[idx * 2 + 1, :] = 0.0
                # L[idx * 2 + 1] = 0.0
                A = A.at[idx * 3 + 1, :].set(0.0)
                L = L.at[idx * 3 + 1].add(0.0)
            if sz:
                # A[idx * 2 + 1, :] = 0.0
                # L[idx * 2 + 1] = 0.0
                A = A.at[idx * 3 + 2, :].set(0.0)
                L = L.at[idx * 3 + 2].add(0.0)

        # self.forces, residuals, rank, s = np.linalg.lstsq(A, -L, rcond=None)
        self.forces = np.linalg.solve(A.T @ A, - A.T @ L)
        # print(rank, 3 * self.N, self.M)

        # residual_threshold = 0.1
        # print(residuals[0])
        # if len(residuals) > 0 and residuals.item() > residual_threshold:
        #     raise ValueError("Nodes could not reach equilibrium!")

        return self.forces
    
    def loss(self, nodes):

        F = self.calculate_forces(nodes)

        lengths = np.zeros_like(F)
        for i, (n1, n2) in enumerate(self.edges):
            # lengths[i] = np.linalg.norm(self.nodes[n1] - self.nodes[n2])
            lengths = lengths.at[i].set(np.linalg.norm(nodes[n1] - nodes[n2]))
    
        weights = np.zeros_like(lengths)

        # beams in tension
        weights = lengths * np.abs(F)

        # compression
        fall_off = 3.0
        # weights[F > 0] *= (lengths[F > 0] + fall_off) / fall_off
        for idx, f in enumerate(F):
            if f > 0:
                weights = weights.at[idx].multiply((lengths[idx] + fall_off)/fall_off)
        # weights = weights.at[F > 0].multiply((lengths[F > 0] + fall_off) / fall_off)

        return np.sum(weights)
    
    def optimize(self, n_frames, lr, save, exp_name):

        if save:
            fig, ax = plt.subplots(subplot_kw={'projection': "3d"})
            frame_folder = f'output/{exp_name}/frames/'
            vid_path = f'output/{exp_name}/animation.mp4'
            if not os.path.exists(frame_folder):
                os.makedirs(frame_folder)

        nodes = self.nodes

        for i in range(n_frames):

            grads = grad(self.loss)(nodes)

            if save:
                plt.cla()
                self.plot(ax, nodes)
                plt.savefig(f"{frame_folder}frame_{i}.png")

            nodes = nodes - lr * self.mask * grads

        if save:
            frames = []
            for i in range(n_frames):
                frames.append(imageio.imread(f"{frame_folder}frame_{i}.png"))
            imageio.mimsave(vid_path, frames, format='mp4', fps=30)
    
    def plot(self, ax, nodes) -> None:

        xs = numpy.array(nodes[:, 0])
        ys = numpy.array(nodes[:, 1])
        zs = numpy.array(nodes[:, 2])
        # draw edges
        for (i, j), f in zip(self.edges, self.forces):
            color = 'r' if f > 0 else 'b'  # red for compression, blue for tension
            segments = [numpy.array([nodes[i, :], nodes[j, :]])]
            line = Line3DCollection(
                segments,
                linewidths=abs(f.primal[0]) + 0.5,
                colors=f'{color}',
                zorder=-1,
                alpha=1.0
            )
            ax.add_collection3d(line)

        # draw loads
        for idx, fx, fy, fz in self.loads:
            fx /= 5
            fy /= 5
            fz /= 5
            a = Arrow3D(
                        [xs[idx] - fx, xs[idx]], [ys[idx] - fy, ys[idx]], [zs[idx] - fz, zs[idx]],
                        mutation_scale=30, 
                        lw=5,
                        arrowstyle="-|>",
                        color="green",
                        alpha=1.0
                    )
            ax.add_artist(a)
            # plt.arrow(xs[idx] - fx, ys[idx] - fy, fx, fy,  # subtract force to put tip at node
            #     head_width = 0.2,
            #     width = 0.05,
            #     color='green',
            #     length_includes_head=True)
            
        # draw anchors
        for idx, x_constrained, y_constrained, z_constrained in self.anchors:
            constrained_size = 0.1
            unconstrained_size = 0.3
            width = constrained_size if x_constrained else unconstrained_size
            height = constrained_size if y_constrained else unconstrained_size
            depth = constrained_size if z_constrained else unconstrained_size

            center = (xs[idx], ys[idx], zs[idx])
            cube = draw_cube(center, (width, height, depth))
            ax.add_collection3d(cube)
            # ax.add_patch(Rectangle(center, width, height, edgecolor="green", facecolor="green"))
        
        # draw nodes
        plt.scatter(xs, ys, zs)
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_zlim([min(nodes[:, 2]), max(nodes[:, 2])])
        ax.set_box_aspect([1, 1, 1])
        # plt.axis('equal')
        # ax.autoscale(enable=True)

nodes = np.array([
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 0.0, 2.0),
    (1.0, 0.0, 2.0),
    (0.0, 1.0, 2.0),
    (1.0, 1.0, 2.0),
    (0.0, 0.0, 3.0),
    (1.0, 0.0, 3.0),
    (0.0, 1.0, 3.0),
    (1.0, 1.0, 3.0)
])
edges = np.array([
    # (0, 1), (0, 2), (1, 3), (2, 3),
    (4, 5), (4, 6), (5, 7), (6, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (8, 9), (8, 10), (9, 11), (10, 11),
    (12, 13), (12, 14), (13, 15), (14, 15),
    (8, 12), (9, 13), (10, 14), (11, 15),
    (4, 8), (5, 9), (7, 11), (6, 10)
])

t = Graph3d(nodes, edges)

t.add_anchor(0)
t.add_anchor(1)
t.add_anchor(2)
t.add_anchor(3)

# t.add_load(4, 1, 2, 10)
# t.add_load(5, 7, -5, 6)
# t.add_load(15, -4, -4, 4)
# t.add_load(11, -4, -4, 4)
# t.add_load(9, -4, -4, 4)

t.add_load(12, 0, 5, 0)
t.add_load(13, 0, 5, 0)

# F = t.calculate_forces(nodes)
# print(F)
# grads = grad(t.loss)(nodes)
# print(grads)

t.optimize(n_frames=200, lr = 0.001, save=True, exp_name="tower")