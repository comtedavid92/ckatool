# CKATool: A Clinical Kinematic Analysis Toolbox for Upper Limb Rehabilitation

Tool for analyzing upper limb movements using 3D motion tracking data.

This repository is a fork of the original CKATool: https://github.com/khaira/ckatool.

## Added in this fork

- Make Rerun an optional dependency, allowing the tool to be used as a library
  for offline kinematic computations
- Modify the jerk computation by computing derivatives per iteration
- Add trunk compensation computation to the neck object

## To use the library

- Download this repository
- Install it using: ```pip install -e .```