#!/usr/bin/env python3

from src.graph2d import Graph2d
import jax.numpy as np

nodes = np.array([
    (0.0, 0.0),
    (0.0, 1.0), 
    (1.0, 0.0),
    (1.0, 1.0), 
    (2.0, 0.0),
    (2.0, 1.0), 
    (3.0, 0.0),
    (3.0, 1.0), 
    (4.0, 0.0),
    (4.0, 1.0), 
])
edges = np.array((
    (0,2), (2,4), (4,6), (6,8),
    (1,3), (3,5), (5,7), (7,9),
    (0,1), (2,3), (4,5), (6,7), (8,9),
    (0,3), (2,5), (5,6), (7,8),
    (1,2),(3,4),(4,7),(6,9)
))

# nodes = np.array([
#     (0.0, 0.0),
#     (0.0, 1.0),
#     (0.0, 2.0),
#     (0.0, 3.0),
#     (1.0, 0.0),
#     (1.0, 1.0),
#     (1.0, 2.0),
#     (1.0, 3.0)
# ])
# edges = np.array((
#     (0, 1), (0, 5), (0, 4),
#     (1, 2), (1, 6), (1, 5), (1, 4),
#     (2, 3), (2, 7), (2, 6), (2, 5),
#     (3, 7), (3, 6),
#     (7, 6),
#     (6, 5),
#     (5, 4)
# ))

t = Graph2d(nodes, edges)
# t.add_load(3, 5, 0)
# t.add_anchor(0)
# t.add_anchor(4)

t.add_load(2, 0, -5)
t.add_load(4, 0, -5)
t.add_load(6, 0, -5)
t.add_anchor(0)
t.add_anchor(8, sx=True)

t.optimize(n_frames=200, lr = 0.0005, save=True, exp_name="beam")