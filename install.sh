#!/bin/bash

cd third_party/marching_cubes
python3 setup.py install

cd ../sparse_octree
python3 setup.py install

cd ../sparse_voxels
python3 setup.py install
