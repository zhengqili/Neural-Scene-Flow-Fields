# Neural-Scene-Flow-Fields
PyTorch implementation of paper "Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes"


## Dependency
The code is tested with Pytorch >= 1.6, the depdenency library includes PIL, opencv, skimage, scipy.

## Video preprocessing 
(1) Download example input videos with SfM poses estimated from COLMAP: https://colmap.github.io/

(2) Download Midas Model from ..., and put it on the directory "nsff_scripts"

(3) Run the following commands to generate required inputs for the models
```bash
    # Usage
    cd nsff_scripts
    # create camera intrinsics/extrinsic format for NeRF
    python save_poses_nerf.py --data_path "/phoenix/S7/zl548/nerf_data/kid-running/dense/"
    # Run single view model
    python run_midas.py --data_path "/home/zl548/nerf_data/kid-running/dense/"
    # RUN optical flow model
    cd RAFT
    python run_flows_video.py --model models/raft-things.pth --data_path /home/zl548/nerf_data/kid-running/dense/ --epi_threhold 1.0
```

## Rendering from pretrained models
(1) Set datadir in config/config_kid-running.txt to the root directory of input video
(2) Fixed Time, Viewpoint Interpolation
```bash
   python run_nerf.py --config configs/config_kid-running.txt --render_bt --target_idx 10
```
(3) Fixed Viewpoint, Time Interpolation
```bash
   python run_nerf.py --config configs/config_kid-running.txt --render_dynamics_slowmo --target_idx 5
```
(3) Space-Time Interpolation
```bash
   python run_nerf.py --config configs/config_kid-running.txt --render_slowmo_bt  --target_idx 10
```

## Training

## Evaluation
