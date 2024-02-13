from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import imageio
import torch
import os

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

class Graph2d:
    def __init__(self, nodes, edges):

        self.N = len(nodes)
        self.M = len(edges)

        self.nodes = torch.tensor(nodes, dtype=torch.float32, requires_grad=True)
        self.edges = edges
        self.loads = []
        self.anchors = []

        self.mask = torch.ones_like(self.nodes)

    def add_load(self, idx: int, fx: float = 0, fy: float = 0):
        self.loads.append((idx, fx, fy))
        self.mask[idx, :] = torch.tensor([0, 0])

    def add_anchor(self, idx: int, sx: bool = True, sy: bool = True):
        self.anchors.append((idx, sx, sy))
        self.mask[idx, :] = torch.tensor([not sx, not sy])

    def proj(self, dx, dy):
        hypot = torch.sqrt(dx ** 2 + dy ** 2)
        return dx / hypot, dy / hypot

    def calculate_forces(self, nodes):

        A = torch.zeros((2 * self.N, self.M))
        for idx, (n1, n2) in enumerate(self.edges):
            n1_x, n1_y = nodes[n1]
            n2_x, n2_y = nodes[n2]
            ux, uy, = self.proj(n1_x - n2_x, n1_y - n2_y)

            # towards N1
            A[n1 * 2][idx], A[n1 * 2 + 1][idx] = ux, uy

            # towards N2
            A[n2 * 2][idx], A[n2 * 2 + 1][idx] = -ux, -uy

        L = torch.zeros((2 * self.N, 1))
        for idx, fx, fy, fz in self.loads:
            L[idx * 2] += fx
            L[idx * 2 + 1] += fy

        for idx, sx, sy in self.anchors:
            if sx:
                A[idx * 2, :] = 0.0
                L[idx * 2] = 0.0
            if sy:
                A[idx * 2 + 1, :] = 0.0
                L[idx * 2 + 1] = 0.0

        self.forces, residuals, rank, s = torch.linalg.lstsq(A, -L, rcond=None, driver='gelsd')

        # residual_threshold = 0.1
        # print(residuals[0])
        # if len(residuals) > 0 and residuals.item() > residual_threshold:
        #     raise ValueError("Nodes could not reach equilibrium!")

        return self.forces
    
    def loss(self, nodes):

        F = self.calculate_forces(nodes)

        lengths = torch.zeros_like(F)
        for i, (n1, n2) in enumerate(self.edges):
            lengths[i] = torch.linalg.norm(self.nodes[n1] - self.nodes[n2])
    
        weights = torch.zeros_like(lengths)

        # beams in tension
        weights = lengths * torch.abs(F)

        # compression
        fall_off = 3.0
        weights[F > 0] *= (lengths[F > 0] + fall_off) / fall_off

        return torch.sum(weights)

        # return torch.sum(torch.abs(F))
    
    def optimize(self, n_frames, lr, save, exp_name):

        if save:
            fig, ax = plt.subplots()
            frame_folder = f'output/{exp_name}/frames/'
            vid_path = f'output/{exp_name}/animation.gif'
            if not os.path.exists(frame_folder):
                os.makedirs(frame_folder)

        print(f"=============================== Starting Experiment: {exp_name} ===============================")
        print(f"Number of iterations: {n_frames}")
        print(f"Learning Rate: {lr}")
        print(f"Plotting: {save}")

        nodes = self.nodes

        pbar = tqdm(range(n_frames))
        # loss = self.loss(nodes)
        for i in pbar:

            loss = self.loss(nodes)
            times = loss.clone().detach().numpy()
            loss = torch.sum(loss)

            if save:
                plt.cla()
                self.plot(ax, nodes, times)
                plt.savefig(f"{frame_folder}frame_{i}.png")

            loss.backward()

            # # Update the value of x using gradient descent
            with torch.no_grad():  # We don't want this operation to be tracked in the computation graph
                nodes -= lr * self.mask * nodes.grad
                
            #     # Manually zero the gradients after updating x
                nodes.grad.zero_()

            pbar.set_description(f'Loss: {loss.item():.4f}')

        if save:
            frames = []
            for i in range(n_frames):
                frames.append(imageio.imread(f"{frame_folder}frame_{i}.png"))
            # imageio.mimsave(vid_path, frames, format='mp4', fps=30)
            imageio.mimsave(vid_path, frames, format='GIF', fps=30, loop=0)
    
    def plot(self, ax, nodes, times) -> None:

        if ax is None:
            fig, ax = plt.subplots()
        plt.axis('equal')

        xs = np.array([node[0].detach() for node in nodes])
        ys = np.array([node[1].detach() for node in nodes])

        # if self.forces is None:
        #     for i, j in self.edges:
        #         plt.plot(xs[[i, j]], ys[[i, j]] , 'k-', zorder=-1)
        # else:
        for (i, j), T in zip(self.edges, times):
            # color = 'r' if f > 0 else 'b'  # red for compression, blue for tension
            color = 'r'
            plt.plot(
                xs[[i, j]], ys[[i, j]], 
                f'{color}-',
                linewidth=abs(T * 20) + 0.1, # still render things of zero force
                zorder=-1)

        for idx, fx, fy in self.loads:
            fx /= 10
            fy /= 10
            plt.arrow(xs[idx] - fx, ys[idx] - fy, fx, fy,  # subtract force to put tip at node
                head_width = 0.2,
                width = 0.05,
                color='green',
                length_includes_head=True)
            
        for idx, x_constrained, y_constrained in self.anchors:
            constrained_size = 0.1
            unconstrained_size = 0.3
            width = constrained_size if x_constrained else unconstrained_size
            height = constrained_size if y_constrained else unconstrained_size

            center = (xs[idx] - width / 2, ys[idx] - height / 2)
            ax.add_patch(Rectangle(center, width, height, edgecolor="green", facecolor="green"))

        plt.scatter(x=xs, y=ys, s=1)