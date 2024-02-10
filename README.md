# structure_optimization

Structure optimization via strain energy minimization. This project is currently work in progress

![Initial Tower Structure](./samples/init_tower.gif)
![Optimal Tower Structure](./samples/optim_tower.gif)

# To Do
- [x] Create a 3D implementation of the original proposal
- [ ] Update Installtion instructions in the README
- [ ] Plot strain energy vs iterations
- [ ] Benchmark with different frameworks and architectures
 - [ ] Implementation in C++
 - [ ] Implementation in Javascript
 - [ ] Create a project page for the project 
- [ ] Implement a learning rate scheduler for greater convergence towards the optimal structure and decrease the wobbliness
- [ ] Maybe include the stiffness matrix into force calculations for more realistic structures
- [ ] Taking inspiration from NEAT(NeuroEvolution of Augmenting Topologies) implement an evolutionary approach for creating and linking nodes to achieve a robust and optimal structure

# Experiments to Perform
- [x] Create a tower and apply two forces on the top
- [ ] Add an additional constraint for volume minimization in the loss function, current hypothesis is that when opimized with gravity the structure will resemble a honeycomb
- [ ] Optimize bridges
- [ ] Create an infinite grid and simulate a hand punching in it

## Useful References
- NEAT (https://ieeexplore.ieee.org/document/6790655)
- Concise summary of NEAT (https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

## Installation

### Creating a virtual environment
