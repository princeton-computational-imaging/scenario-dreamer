# Official Repository for Scenario Dreamer

<p align="left">
<a href="https://arxiv.org/abs/2503.22496" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2503.22496-b31b1b.svg?style=flat" /></a>
<a href="https://princeton-computational-imaging.github.io/scenario-dreamer/" alt="webpage">
    <img src="https://img.shields.io/badge/Project Page-Scenario Dreamer-blue" /></a>
<a href="https://paperswithcode.com/paper/scenario-dreamer-vectorized-latent-diffusion">
    <img alt="Static Badge" src="https://img.shields.io/badge/paper_with_code-link-turquoise?logo=paperswithcode" /></a>

> [**Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments**](https://arxiv.org/abs/2503.22496)  <br>
> [Luke Rowe](https://rluke22.github.io)<sup>1,2,6</sup>, [Roger Girgis](https://mila.quebec/en/person/roger-girgis/)<sup>1,3,6</sup>, [Anthony Gosselin](https://www.linkedin.com/in/anthony-gosselin-098b7a1a1/)<sup>1,3</sup>, [Liam Paull](https://liampaull.ca/)<sup>1,2,5</sup>, [Christopher Pal](https://sites.google.com/view/christopher-pal)<sup>1,2,3,5</sup>, [Felix Heide](https://www.cs.princeton.edu/~fheide/)<sup>4,6</sup>  <br>
> <sup>1</sup> Mila, <sup>2</sup> Université de Montréal, <sup>3</sup> Polytechnique Montréal, <sup>4</sup> Princeton University, <sup>5</sup> CIFAR AI Chair, <sup>6</sup> Torc Robotics <br>
> <br>
> Computer Vision and Pattern Recognition (CVPR), 2025 <br>
>

We propose Scenario Dreamer, a fully data-driven closed-loop generative simulator for autonomous vehicle planning.

<video src="https://github.com/user-attachments/assets/83bcea5f-a459-45b7-8d36-eb9dd76e100a" width="250" height="250"></video>

## Repository Timeline

- [x] [06/11/2025] Environment setup
- [x] [06/11/2025] Dataset Preprocessing
- [ ] [ETA: 06/27/2025] Train Scenario Dreamer autoencoder model on Waymo
- [ ] [ETA: 06/27/2025] Train Scenario Dreamer latent diffusion model on Waymo
- [ ] [ETA: 06/27/2025] Support inpainting and lane-conditioned object generation modes on Waymo
- [ ] [ETA: 06/27/2025] Support visualization of Scenario Dreamer initial scenes
- [ ] [ETA: 07/05/2025] Support evaluation of Scenario Dreamer model on Waymo
- [ ] [ETA: 07/05/2025] Compatibility with nuPlan dataset
- [ ] [ETA: 07/05/2025] Release of pre-trained Scenario Dreamer models on Waymo and nuPlan
- [ ] [ETA: 07/05/2025] Train CtRL-Sim behaviour model on Waymo
- [ ] [ETA: 07/05/2025] Train Scenario-Dreamer compatible agents in GPUDrive
- [ ] [ETA: 07/05/2025] Evaluate planners in Scenario Dreamer environments
- [ ] [ETA: 07/05/2025] SLEDGE and DriveSceneGen baseline reproduction and evaluation

## Setup

Start by cloning the repository
```
git clone https://github.com/RLuke22/scenario-dreamer-waymo
cd scenario-dreamer-waymo
```

This repository assumes you have a "scratch" directory for larger files (datasets, checkpoints, etc.). If disk space is not an issue, you can keep everything in the repository directory:
```
export SCRATCH_ROOT=$(pwd) # prefer a separate drive? Point SCRATCH_ROOT there instead.
```

Define environment variables to let the code know where things live:
```
bash scripts/define_env_variables.sh
```

### Conda Setup 

```
# create conda environment
conda env create -f environment.yml
conda activate scenario-dreamer
```

## Waymo Dataset Preparation

Download the Waymo Open Motion Dataset (v1.1.0) into your scratch directory with the following directory structure:

```
$SCRATCH_ROOT/waymo_open_dataset_motion_v_1_1_0/
├── training/
│   ├── training.tfrecord-00000-of-01000
│   ├── …
│   └── training.tfrecord-00999-of-01000
├── validation/
│   ├── validation.tfrecord-00000-of-00150
│   ├── …
│   └── validation.tfrecord-00149-of-00150
└── testing/
    ├── testing.tfrecord-00000-of-00150
    ├── …
    └── testing.tfrecord-00149-of-00150
```

Then, we preprocess the waymo dataset to prepare for Scenario Dreamer model training:
```
bash scripts/extract_data.sh # extract relevant data from tfrecords and create train/val/test splits
bash scripts/preprocess_waymo_dataset.sh # preprocess data to facilitate efficient model training
```


## Citation

```bibtex
@InProceedings{rowe2025scenariodreamer,
  title={Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments},
  author={Rowe, Luke and Girgis, Roger and Gosselin, Anthony and Paull, Liam and Pal, Christopher and Heide, Felix},
  booktitle = {CVPR},
  year={2025}
}
```
