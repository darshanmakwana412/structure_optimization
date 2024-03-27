#!/bin/bash
docker run -it --rm --detach-keys="ctrl-a" --name=structure_optimization-container --gpus '"device=0"' --ipc=host -p 8888:8888 -v "/home/darshan/Academics/Sem 6/ME 312/structure_optimization:/workspace" -v "/home/darshan/Academics/Sem 6/ME 312/structure_optimization/jupyter.sh:/workspace/jupyter.sh" -u 1000:1000 pytorch:23.02-py3
