# Neural-Scene-Flow-Fields
PyTorch implementation of paper "Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes"


## Dependency
The code is tested with Pytorch >= 1.6, the depdenency library includes PIL, opencv, skimage, scipy, cupy, imageio.

## Video preprocessing 
(1) Download nerf_data.zip from https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing, an example input videos with SfM poses estimated from COLMAP: https://colmap.github.io/

(2) Download single view depth prediction model "model.pt" from https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing, and put it on the folder "nsff_scripts".

(3) Run the following commands to generate required inputs:
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
(1) Download pretraind model "kid-running_ndc_5f_sv_of_sm_unify3_F00-30.zip" from https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing, and unzip and put it in the folder "nsff_exp/logs/kid-running_ndc_5f_sv_of_sm_unify3_F00-30/360000.tar". Set datadir in config/config_kid-running.txt to the root directory of input video. Then go to directory "nsff_exp":
```bash
   cd nsff_exp
```
(2) Rendering with fixed time, viewpoint interpolation
```bash
   python run_nerf.py --config configs/config_kid-running.txt --render_bt --target_idx 10
```
(3) Rendering with fixed Viewpoint, time interpolation
```bash
   python run_nerf.py --config configs/config_kid-running.txt --render_dynamics_slowmo --target_idx 5
```
(3) Rendering with space-time interpolation
```bash
   python run_nerf.py --config configs/config_kid-running.txt --render_slowmo_bt  --target_idx 10
```

## Training
(1) In configs/config_kid-running.txt, changing expname to any name that you like (different from the original one), and running the following command:
```bash
    python run_nerf.py --config configs/config_kid-running.txt
```
The per-scene training takes ~2 days using 2 Nvidia V100 GPUs.

## Evaluation on Dynamic Scene Dataset
(1) Download Dynamic Scene dataset "dynamic_scene_data_full.zip" from https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing

(2) Download pretrained models "dynamic_scene_pretrained_models.zip" from https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing, unzip and put them in the folder "nsff_exp/logs/" 

(3) Run the following command for each scene:
```bash
   # Usage: configs/config_xxx.txt indicates each scene name such as config_balloon1-2.txt in nsff/configs
   python evaluation.py --config configs/config_xxx.txt
```

