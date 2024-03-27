#!/usr/bin/env python3

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import imageio
import torch
import os

'''
To DO:
    - [ ] Use Pytorch optimizer for updating the nodes
    - [ ] Implement a learning rate scheduler
    - [ ] Optimize the lr scheduler
'''

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

class Graph2d:
    def __init__(self, nodes, edges):

        self.N = len(nodes)
        self.M = len(edges)

        self.original_nodes = [torch.tensor(node, requires_grad=False, dtype=torch.float32) for node in nodes]
        self.nodes = [torch.tensor(node, requires_grad=True, dtype=torch.float32) for node in nodes]
        self.edges = edges
        self.loads = []
        self.anchors = []

    def add_load(self, idx: int, fx: float = 0, fy: float = 0):
        self.loads.append((idx, fx, fy))
        self.nodes[idx].requires_grad = True

    def add_anchor(self, idx: int, sx: bool = True, sy: bool = True):
        self.anchors.append((idx, sx, sy))
        self.nodes[idx].requires_grad = False

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
        for idx, fx, fy in self.loads:
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

    def norm(self, n1, n2):
        return torch.linalg.norm(n1 - n2)
    
    def loss(self, nodes):

        F = self.calculate_forces(nodes)

        loss = torch.sum(torch.abs(F))

        lengths = torch.zeros_like(F)
        for i, (n1, n2) in enumerate(self.edges):
            lengths[i] = torch.linalg.norm(self.nodes[n1] - self.nodes[n2])
    
        weights = torch.zeros_like(lengths)

        # beams in tension
        weights = lengths * torch.abs(F)

        # compression
        fall_off = 3.0
        weights[F > 0] *= (lengths[F > 0] + fall_off) / fall_off

        loss = torch.sum(weights)

        # Don't want the anchor points to wobble a lot
        for idx, fx, fy in self.loads:
            loss += 100 * self.norm(nodes[idx], self.original_nodes[idx])

        for idx, sx, sy in self.anchors:
            loss += 100 * self.norm(nodes[idx], self.original_nodes[idx])

        return loss
    
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

        # Define the optimizer
        params = [n for n in self.nodes if n.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        # Define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

        pbar = tqdm(range(n_frames))
        for i in pbar:

            optimizer.zero_grad()  # Zero the gradients at the start of each loop
            loss = self.loss(nodes)
            loss = torch.sum(loss)

            if save:
                plt.cla()
                self.plot(ax, nodes)
                plt.savefig(f"{frame_folder}frame_{i}.png")

            loss.backward()

            # Update the nodes using the optimizer
            optimizer.step()

            # Apply the learning rate scheduler
            scheduler.step()

            # Update tqdm description with loss and current learning rate
            current_lr = scheduler.get_last_lr()[0]  # Get the last learning rate
            pbar.set_description(f'Loss: {loss.item():.4f} LR: {current_lr:.5f}')

        if save:
            frames = []
            for i in range(n_frames):
                frames.append(imageio.imread(f"{frame_folder}frame_{i}.png"))
            # imageio.mimsave(vid_path, frames, format='mp4', fps=30)
            imageio.mimsave(vid_path, frames, format='GIF', fps=30, loop=0)
    
    def plot(self, ax, nodes) -> None:

        if ax is None:
            fig, ax = plt.subplots()
        plt.axis('equal')

        xs = np.array([node[0].detach() for node in nodes])
        ys = np.array([node[1].detach() for node in nodes])

        # if self.forces is None:
        #     for i, j in self.edges:
        #         plt.plot(xs[[i, j]], ys[[i, j]] , 'k-', zorder=-1)
        # else:

        for (i, j), F in zip(self.edges, self.forces.detach().numpy()):
            color = 'r' if F > 0 else 'b'  # red for compression, blue for tension
            # color = 'r'
            plt.plot(
                xs[[i, j]], ys[[i, j]], 
                f'{color}-',
                linewidth=abs(F * 2) + 0.1, # still render things of zero force
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

# nodes = np.array([
#     (0.0, 0.0),
#     (0.0, 1.0), 
#     (1.0, 0.0),
#     (1.0, 1.0), 
#     (2.0, 0.0),
#     (2.0, 1.0), 
#     (3.0, 0.0),
#     (3.0, 1.0), 
#     (4.0, 0.0),
#     (4.0, 1.0), 
#     (0.0, 2.0),
#     (1.0, 2.0),
#     (2.0, 2.0),
#     (3.0, 2.0),
#     (4.0, 2.0)
# ])
# edges = np.array((
#     (0,2), (2,4), (4,6), (6,8),
#     (1,3), (3,5), (5,7), (7,9),
#     (0,1), (2,3), (4,5), (6,7), (8,9),
#     (0,3), (2,5), (5,6), (7,8),
#     (1,2),(3,4),(4,7),(6,9),
#     (1, 10), (3, 11), (5, 12), (7, 13), (9, 14),
#     (10, 11), (11, 12), (12, 13), (13, 14),
#     (10, 3), (11, 1), (12, 7), (13, 5), (14, 7), (13, 9), (3, 12), (5, 11)
# ))

# nodes = np.array([
#     (0.0, 0.0),
#     (1.0, 0.0),
#     (2.0, 0.0),
#     (3.0, 0.0),
#     (4.0, 0.0),
#     (0.0, 1.0)
# ])
# edges = []
# for i in range(len(nodes) - 1):
#     edges.append((i, 5))
# edges = np.array(edges)

# t = Graph2d(nodes, edges)

# t.add_load(2, 0, -5)
# t.add_load(4, 0, -5)
# t.add_load(6, 0, -5)

# t.add_anchor(0)
# t.add_anchor(8)

# t.optimize(n_frames=200, lr = 0.005, save=True, exp_name="beam")

nodes = np.array([
    (0.0, 0.0),
    (0.0, 1.0), 
    (1.0, 0.0),
    (1.0, 1.0), 
    (2.0, 0.0),
    # (2.0, 1.0), 
    (3.0, 0.0),
    (3.0, 1.0), 
    (4.0, 0.0),
    (4.0, 1.0),
])

edges = []
for idx1, n1 in enumerate(nodes):
    for idx2, n2 in enumerate(nodes):
        if not (n1[0] == n2[0] and n1[1] == n2[1]):
            edges.append((idx1, idx2))
edges = np.array(edges)

t = Graph2d(nodes, edges)

t.add_load(2, 0, -5)
t.add_load(4, 0, -5)
t.add_load(5, 0, -5)

t.add_anchor(0)
t.add_anchor(7)

t.optimize(n_frames=200, lr = 0.005, save=True, exp_name="nt_1")

class Genome:
    def __init__(self, genome_idx: int) -> None:
        self.genome_idx = genome_idx
        self.num_nodes = 0
        self.nodes = {}
        self.edges = []

    def add_node(self, x: float, y: float) -> None:
        # Check if the node is already in the genome
        if (x, y) not in self.nodes.values():
            idx = self.num_nodes
            self.nodes[idx] = (x, y)
            self.num_nodes += 1
        else:
            print(f"Adding node ({x}, {y}) to genome {self.genome_idx} would create a duplicate node!")

    def add_edge(self, idx1: int, idx2: int) -> None:
        self.edges.append((idx1, idx2))

# g1 = Genome(0)
# g1.add_node(0, 0)
# g1.add_node(0, 1)
# g1.add_node(1, 0)
# g1.add_node(1, 1)
# g1.add_node(2, 0)
# g1.add_node(2, 1)
# g1.add_node(3, 0)
# g1.add_node(3, 1)
# g1.add_node(4, 0)
# g1.add_node(4, 1)
# g1.add_node(0, 0)

# g1.add_edge(0, 1)
# g1.add_edge(1, 2)
# g1.add_edge(2, 3)
# g1.add_edge(3, 4)
# g1.add_edge(4, 5)
# g1.add_edge(5, 6)
# g1.add_edge(6, 7)
# g1.add_edge(7, 8)
# g1.add_edge(8, 9)
# g1.add_edge(9, 0)

# print(g1.nodes)
# print(g1.edges)
