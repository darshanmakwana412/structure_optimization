#!/usr/bin/env python3

from src.graph2d import Graph2d
import numpy as np
import torch

class Curve(Graph2d):
    def loss(self, nodes):

        T = torch.zeros(len(nodes) - 1)
        g = 1

        for i in range(len(nodes) - 1):
            # n1x, n1y = nodes[i]
            # n2x, n2y = nodes[i+1]
            # ux, uy = self.proj(n2x - n1x, n2y - n1y)
            l = torch.linalg.norm(nodes[i] - nodes[i+1])
            v = torch.sqrt(2 * g * (nodes[0, 1] - (nodes[i, 1] + nodes[i+1, 1])/2))
            T[i] = l / v
            # v1 = torch.sqrt( 2 * g * (nodes[0, 1] - nodes[i, 1] ))
            # T[i] = (v1 - torch.square(torch.square(v1) - 2 * g * uy * l)) / (g * uy)

        return T

N = 70
N1 = (0, 2)
N2 = (3, 0)
nodes = []
edges = []

for i in range(N):
    nodes.append([n1 + (i / N) * (n2 - n1) for n1, n2 in zip(N1, N2)])
    edges.append((i, i + 1))

nodes.append(N2)


nodes = np.array(nodes, dtype=np.float64)
edges = np.array(edges)

t = Curve(nodes, edges)

t.add_anchor(0)
t.add_anchor(N)

t.optimize(n_frames=1000, lr = 0.1, save=True, exp_name="brachistochrone")