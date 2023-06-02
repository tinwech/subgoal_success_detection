# Benchmarking Spatial Relation Modeling for Subgoal Success Detection in Robotic Rearrangement

This repository contains the code for benchmarking spatial relation modeling in the context of subgoal success detection in robotic rearrangement tasks. The goal of this research is to evaluate the effectiveness of different spatial relation modeling techniques in predicting the success of subgoals during robotic rearrangement tasks.

## Introduction

In this project, we aim to address the problem of subgoal success detection in robotic rearrangement tasks. Robotic rearrangement involves moving objects from an initial configuration to a target configuration through a series of subgoals. Successful completion of each subgoal is crucial to achieve the overall task objective. We explore various spatial relation modeling techniques to predict the success or failure of subgoals.

![Picture1](https://github.com/tinwech/subgoal_success_detection/assets/80531783/af530e8e-5d35-40a0-bf9a-2a88248c5618)

## Data

The data of success and failed scene are collected from the [Ravens](https://github.com/google-research/ravens) environment.

## Models

We implement and compare multiple spatial relation modeling techniques for subgoal success detection. The following models are included in this repository:

- `baseline.py`: The 3D convolution baseline model.
- `baseline_roi_model.py`: ROI spatial relation modeling.
- `scene_graph_model.py`: Scene graph spatial relation modeling.
- `CPN_model.py`: Contact Point Network spatial relation modeling.
- `histogram_model.py`: Histogram spatial relation modeling.

## Getting started

### Environment setup

```sh
conda env create -f env.yml
```

### Preprocessing

Before training the models, it is necessary to preprocess the dataset. Follow the steps below to preprocess the dataset:

1. `trans_rgbd_to_voxel.py`: Convert RGB-D image into voxel grids.
2. `get_ROI_CPN.py`: Get ROI proposals and CPN data from the voxel grids.

### Train and Inference

```sh
python main.py [-m <model_name>]
```
