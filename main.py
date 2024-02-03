#!/usr/bin/env python3

from src.graph2d import Graph2d
import jax.numpy as np

nodes = np.array([
    (0.0, 0.0),
    (0.0, 1.0),
    (0.0, 2.0),
    (0.0, 3.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (1.0, 2.0),
    (1.0, 3.0)
])
edges = np.array((
    (0, 1), (0, 5), (0, 4),
    (1, 2), (1, 6), (1, 5), (1, 4),
    (2, 3), (2, 7), (2, 6), (2, 5),
    (3, 7), (3, 6),
    (7, 6),
    (6, 5),
    (5, 4)
))

t = Graph2d(nodes, edges)
t.add_load(3, 5, 0)
t.add_anchor(0)
t.add_anchor(4)

t.optimize(n_frames=200, lr = 1, save=True, exp_name="cantilever")