from mpl_toolkits.mplot3d.art3d import Line3DCollection
from utils import draw_cube, Arrow3D
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

class Graph3d:
    def __init__(self, nodes, edges):

        self.N = len(nodes)
        self.M = len(edges)

        self.nodes = torch.tensor(nodes, dtype=torch.float32, requires_grad=True)
        self.edges = edges
        self.loads = []
        self.anchors = []

        self.mask = torch.ones_like(self.nodes)

    def add_load(self, idx: int, fx: float = 0, fy: float = 0, fz: float = 0):
        self.loads.append((idx, fx, fy, fz))
        self.mask[idx, :] = torch.tensor([0, 0, 0])

    def add_anchor(self, idx: int, sx: bool = True, sy: bool = True, sz: bool = True):
        self.anchors.append((idx, sx, sy, sz))
        self.mask[idx, :] = torch.tensor([not sx, not sy, not sz])

    def proj(self, dx, dy, dz):
        hypot = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return dx / hypot, dy / hypot, dz / hypot

    def calculate_forces(self, nodes):

        A = torch.zeros((3 * self.N, self.M))
        for idx, (n1, n2) in enumerate(self.edges):
            n1_x, n1_y, n1_z = nodes[n1]
            n2_x, n2_y, n2_z = nodes[n2]
            ux, uy, uz = self.proj(n1_x - n2_x, n1_y - n2_y, n1_z - n2_z)

            # towards N1
            A[n1 * 3][idx], A[n1 * 3 + 1][idx], A[n1 * 3 + 2][idx] = ux, uy, uz

            # towards N2
            A[n2 * 3][idx], A[n2 * 3 + 1][idx], A[n2 * 3 + 2][idx] = -ux, -uy, -uz

        L = torch.zeros((3 * self.N, 1))
        for idx, fx, fy, fz in self.loads:
            L[idx * 3] += fx
            L[idx * 3 + 1] += fy
            L[idx * 3 + 2] += fz

        for idx, sx, sy, sz in self.anchors:
            if sx:
                A[idx * 3, :] = 0.0
                L[idx * 3] = 0.0
            if sy:
                A[idx * 3 + 1, :] = 0.0
                L[idx * 3 + 1] = 0.0
            if sz:
                A[idx * 3 + 2, :] = 0.0
                L[idx * 3 + 2] = 0.0

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
    
    def optimize(self, n_frames, lr, save, exp_name):

        if save:
            fig, ax = plt.subplots(subplot_kw={'projection': "3d"})
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

            if save:
                plt.cla()
                self.plot(ax, nodes, i)
                plt.savefig(f"{frame_folder}frame_{i}.png")

            loss.backward()

            # Update the value of x using gradient descent
            with torch.no_grad():  # We don't want this operation to be tracked in the computation graph
                nodes -= lr * self.mask * nodes.grad
                
                # Manually zero the gradients after updating x
                nodes.grad.zero_()

            pbar.set_description(f'Loss: {loss.item():.4f}')

        if save:
            frames = []
            for i in range(n_frames):
                frames.append(imageio.imread(f"{frame_folder}frame_{i}.png"))
            # imageio.mimsave(vid_path, frames, format='mp4', fps=30)
            imageio.mimsave(vid_path, frames, format='GIF', fps=30, loop=0)
    
    def plot(self, ax, nodes, azim) -> None:

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': "3d"})

        xs = np.array([nodes[0].detach() for nodes in self.nodes])
        ys = np.array([nodes[1].detach() for nodes in self.nodes])
        zs = np.array([nodes[2].detach() for nodes in self.nodes])
        # draw edges
        for (i, j), f in zip(self.edges, self.forces.detach().numpy()):
            color = 'r' if f > 0 else 'b'  # red for compression, blue for tension
            segments = [np.array([[xs[i], ys[i], zs[i]], [xs[j], ys[j], zs[j]]])]
            line = Line3DCollection(
                segments,
                linewidths=abs(f[0]/1.2) + 0.1,
                colors=f'{color}',
                zorder=-1,
                alpha=1.0
            )
            ax.add_collection3d(line)

        # draw loads
        for idx, fx, fy, fz in self.loads:
            fx /= 8
            fy /= 8
            fz /= 8
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
        # ax.autoscale(enable=True, axis='both', tight=True)
        buffer_percent = 0.1

        # Calculate buffer space for each axis
        x_buffer = (np.max(xs) - np.min(xs)) * buffer_percent
        y_buffer = (np.max(ys) - np.min(ys)) * buffer_percent
        z_buffer = (np.max(zs) - np.min(zs)) * buffer_percent

        # Adjust axis limits with buffer space
        x_min, x_max = np.min(xs) - x_buffer, np.max(xs) + x_buffer
        y_min, y_max = np.min(ys) - y_buffer, np.max(ys) + y_buffer
        z_min, z_max = np.min(zs) - z_buffer, np.max(zs) + z_buffer

        # Set axis limits
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        ax.set_box_aspect([np.ptp(xs), np.ptp(ys), np.ptp(zs)])
        ax.grid(False)
        ax.view_init(elev=30, azim=azim)
        ax.text(0.02, 0.98, 0.98, f'Iteration {azim}', transform=ax.transAxes, color='black', fontsize=12, horizontalalignment='left', verticalalignment='top')
        # ax.set_zlim([min(nodes[:, 2]), max(nodes[:, 2])])
        # ax.set_box_aspect([1, 1, 1])
        # plt.axis('equal')
        # ax.autoscale(enable=True)

# Tower

# nodes = np.array([
#     (0.0, 0.0, 0.0),
#     (1.0, 0.0, 0.0),
#     (0.0, 1.0, 0.0),
#     (1.0, 1.0, 0.0),
#     (0.0, 0.0, 1.0),
#     (1.0, 0.0, 1.0),
#     (0.0, 1.0, 1.0),
#     (1.0, 1.0, 1.0),
#     (0.0, 0.0, 2.0),
#     (1.0, 0.0, 2.0),
#     (0.0, 1.0, 2.0),
#     (1.0, 1.0, 2.0),
#     (0.0, 0.0, 3.0),
#     (1.0, 0.0, 3.0),
#     (0.0, 1.0, 3.0),
#     (1.0, 1.0, 3.0)
# ])
# edges = [
#     # (0, 1), (0, 2), (1, 3), (2, 3),
#     (4, 5), (4, 6), (5, 7), (6, 7),
#     (0, 4), (1, 5), (2, 6), (3, 7),
#     (8, 9), (8, 10), (9, 11), (10, 11),
#     (12, 13), (12, 14), (13, 15), (14, 15),
#     (8, 12), (9, 13), (10, 14), (11, 15),
#     (4, 8), (5, 9), (7, 11), (6, 10),
#     (0, 5), (2, 4), (1, 4), (0, 6), (1, 7), (5, 3), (6, 3), (7, 2)
# ]
# cross = [(0, 5), (2, 4), (1, 4), (0, 6), (1, 7), (5, 3), (6, 3), (7, 2)]
# for (ci, cj) in cross:
#     edges.append((ci + 4, cj + 4))
# for (ci, cj) in cross:
#     edges.append((ci + 8, cj + 8))
# edges.append((12, 15))
# edges.append((13, 14))
# edges.append((8, 11))
# edges.append((9, 10))
# edges.append((4, 7))
# edges.append((5, 6))
# edges = np.array(edges)

# t = Graph3d(nodes, edges)

# t.add_anchor(0)
# t.add_anchor(1)
# t.add_anchor(2)
# t.add_anchor(3)

# # t.add_load(4, 1, 2, 10)
# # t.add_load(5, 7, -5, 6)
# # t.add_load(15, -4, -4, 4)
# # t.add_load(11, -4, -4, 4)
# # t.add_load(9, -4, -4, 4)

# t.add_load(12, 5, 5, 0)
# t.add_load(13, -5, 5, 0)

# # F = t.calculate_forces(nodes)
# # print(F)
# # grads = grad(t.loss)(nodes)
# # print(grads)

# t.optimize(n_frames=360, lr = 0.001, save=True, exp_name="tower")
        
# Bridge
        
nodes = []
for i in range(5):
    nodes.append([0, i, 0])
    nodes.append([0, i, 1])
    nodes.append([1, i, 0])
    nodes.append([1, i, 1])
    
edges = []
for i in range(5):
    edges.extend([[4 * i + 0, 4 *i + 1], [4 * i + 0, 4 *i + 2], [4 * i + 1, 4 *i + 3], [4 * i + 3, 4 *i + 2], [4 * i + 0, 4 *i + 3], [4 * i + 2, 4 *i + 1]])
for i in range(1, 5):
    for j in range(4):
        edges.extend([[4 * (i-1) + j, 4 * i + j]])
    for (u1, u2) in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        edges.extend([[4 * (i-1) + u1, 4 * i + u2]])
        edges.extend([[4 * (i-1) + u2, 4 * i + u1]])

nodes = np.array(nodes)
edges = np.array(edges)

t = Graph3d(nodes, edges)

t.add_anchor(0)
t.add_anchor(2)
t.add_anchor(16)
t.add_anchor(18)

t.add_load(4, 0, 0, 5)
t.add_load(6, 0, 0, 5)
t.add_load(8, 0, 0, 5)
t.add_load(10, 0, 0, 5)
t.add_load(12, 0, 0, 5)
t.add_load(14, 0, 0, 5)

t.optimize(n_frames=360, lr = 0.001, save=True, exp_name="bridge")