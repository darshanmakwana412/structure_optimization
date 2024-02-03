from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import grad
import imageio
import numpy
import os

class Graph2d:
    def __init__(self, nodes, edges):
        self.N = len(nodes)
        self.M = len(edges)

        self.nodes = nodes
        self.edges = edges
        self.loads = []
        self.anchors = []

        self.mask = numpy.ones_like(self.nodes)

    def add_load(self, idx: int, fx: float = 0, fy: float = 0):
        self.loads.append((idx, fx, fy))
        self.mask[idx, :] = [0, 0]

    def add_anchor(self, idx: int, sx: bool = True, sy: bool = True):
        self.anchors.append((idx, sx, sy))
        self.mask[idx, :] = [not sx, not sy]

    def proj(self, dx, dy):
        hypot = np.sqrt(dx ** 2 + dy ** 2)
        return dx / hypot, dy / hypot

    def calculate_forces(self, nodes):

        A = np.zeros((2 * self.N, self.M))
        for idx, (n1, n2) in enumerate(self.edges):
            n1_x, n1_y = nodes[n1]
            n2_x, n2_y = nodes[n2]
            ux, uy = self.proj(n1_x - n2_x, n1_y - n2_y)

            # # towards N1
            # A[n1 * 2][idx], A[n1 * 2 + 1][idx] = ux, uy

            # # towards N2
            # A[n2 * 2][idx], A[n2 * 2 + 1][idx] = -ux, -uy

            A = A.at[n1 * 2, idx].set(ux)
            A = A.at[n1 * 2 + 1, idx].set(uy)
            A = A.at[n2 * 2, idx].set(-ux)
            A = A.at[n2 * 2 + 1, idx].set(-uy)


        L = np.zeros((2 * self.N, 1))
        for idx, fx, fy in self.loads:
            # L[idx * 2] += fx
            # L[idx * 2 + 1] += fy

            L = L.at[idx * 2].add(fx)
            L = L.at[idx * 2 + 1].add(fy)

        for idx, sx, sy in self.anchors:
            if sx:
                # A[idx * 2, :] = 0.0
                # L[idx * 2] = 0.0
                A = A.at[idx * 2, :].set(0.0)
                L = L.at[idx * 2].add(0.0)
            if sy:
                # A[idx * 2 + 1, :] = 0.0
                # L[idx * 2 + 1] = 0.0
                A = A.at[idx * 2 + 1, :].set(0.0)
                L = L.at[idx * 2 + 1].add(0.0)

        self.forces, residuals, rank, s = np.linalg.lstsq(A, -L, rcond=None)

        residual_threshold = 0.1
        if len(residuals) > 0 and residuals.item() > residual_threshold:
            raise ValueError("Nodes could not reach equilibrium!")

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
            fig, ax = plt.subplots()
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
            
            if save:
                plt.savefig(f"{frame_folder}frame_{i}.png")

            nodes = nodes - lr * self.mask * grads

        if save:
            frames = []
            for i in range(n_frames):
                frames.append(imageio.imread(f"{frame_folder}frame_{i}.png"))
            imageio.mimsave(vid_path, frames, format='mp4', fps=30)
    
    def plot(self, ax, nodes) -> None:
        if ax is None:
            fig, ax = plt.subplots()
        plt.axis('equal')

        xs = numpy.array(nodes[:, 0])
        ys = numpy.array(nodes[:, 1])
        
        # draw edges

        if self.forces is None:
            for i, j in self.edges:
                plt.plot(xs[[i, j]], ys[[i, j]] , 'k-', zorder=-1)
        else:
            for (i, j), f in zip(self.edges, self.forces):
                color = 'r' if f > 0 else 'b'  # red for compression, blue for tension
                plt.plot(
                    xs[[i, j]], ys[[i, j]], 
                    f'{color}-',
                    linewidth=abs(f.primal[0]) + 0.1, # still render things of zero force
                    zorder=-1)

        # draw loads
        for idx, fx, fy in self.loads:
            fx /= 10
            fy /= 10
            plt.arrow(xs[idx] - fx, ys[idx] - fy, fx, fy,  # subtract force to put tip at node
                head_width = 0.2,
                width = 0.05,
                color='green',
                length_includes_head=True)
            
        # draw anchors
        for idx, x_constrained, y_constrained in self.anchors:
            constrained_size = 0.1
            unconstrained_size = 0.3
            width = constrained_size if x_constrained else unconstrained_size
            height = constrained_size if y_constrained else unconstrained_size

            center = (xs[idx] - width / 2, ys[idx] - height / 2)
            ax.add_patch(Rectangle(center, width, height, edgecolor="green", facecolor="green"))
        
        # draw nodes
        plt.scatter(xs, ys)