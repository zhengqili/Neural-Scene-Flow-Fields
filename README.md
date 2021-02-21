# Neural-Scene-Flow-Fields
PyTorch implementation of paper "Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes"


## Dependency
The code is tested with Python3, Pytorch >= 1.6 and CUDA >= 10.2, the dependencies includes configargparse, numpy, PIL, matplotlib, opencv, scikit-image, scipy, cupy, imageio.

## Video preprocessing 
(1) Download nerf_data.zip from https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing, an example input video with SfM camera poses and intrinsics estimated from COLMAP: https://colmap.github.io/ (Note you need to use COLMAP "colmap image_undistorter" command to undistort input images to get "dense" folder as shown in the example, this dense folder should include "images" and "sparse" folder used for preprocessing).

(2) Download single view depth prediction model "model.pt" from https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing, and put it on the folder "nsff_scripts".

(3) Run the following commands to generate required inputs for training/inference:
```bash
    # Usage
    cd nsff_scripts
    # create camera intrinsics/extrinsic format for NSFF, same as original NeRF where it uses imgs2poses.py script from the LLFF code: https://github.com/Fyusion/LLFF/blob/master/imgs2poses.py
    python save_poses_nerf.py --data_path "/home/xxx/Neural-Scene-Flow-Fields/kid-running/dense/"
    # Resize input images and run single view model
    python run_midas.py --data_path "/home/xxx/Neural-Scene-Flow-Fields/kid-running/dense/"
    # Run optical flow model 
    cd RAFT
    ./download_models.sh
    python run_flows_video.py --model models/raft-things.pth --data_path /home/xxx/Neural-Scene-Flow-Fields/kid-running/dense/ --epi_threhold 1.0
```

## Rendering from an example pretrained model
(1) Download pretraind model "kid-running_ndc_5f_sv_of_sm_unify3_F00-30.zip" from https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing. Unzipping and putting it in the folder "nsff_exp/logs/kid-running_ndc_5f_sv_of_sm_unify3_F00-30/360000.tar". 

Set datadir in config/config_kid-running.txt to the root directory of input video. Then go to directory "nsff_exp":
```bash
   cd nsff_exp
```

(2) Rendering with fixed time, viewpoint interpolation
```bash
   python run_nerf.py --config configs/config_kid-running.txt --render_bt --target_idx 10
```

By running the example command, you should get the following result:
![Alt Text](https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function/blob/master/demo/ours_34558526690_e5ba5b3b9d.jpg.gif)

(3) Rendering with fixed viewpoint, time interpolation
```bash
   python run_nerf.py --config configs/config_kid-running.txt --render_lockcam_slowmo --target_idx 5
```

By running the example command, you should get the following result:
![Alt Text](https://github.com/zhengqili/Neural-Scene-Flow-Fields/blob/master/demo/vi.gif)

(3) Rendering with space-time interpolation
```bash
   python run_nerf.py --config configs/config_kid-running.txt --render_slowmo_bt  --target_idx 10
```

By running the example command, you should get the following result:
![Alt Text](https://github.com/zhengqili/Neural-Scene-Flow-Fields/blob/master/demo/vi.gif)

## Training
(1) In configs/config_kid-running.txt, modifying expname to any name you like (different from the original one), and running the following command for training the model:
```bash
    python run_nerf.py --config configs/config_kid-running.txt
```
The per-scene training takes ~2 days using 2 Nvidia V100 GPUs.

## Evaluation on the Dynamic Scene Dataset
(1) Download Dynamic Scene dataset "dynamic_scene_data_full.zip" from https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing

(2) Download pretrained model "dynamic_scene_pretrained_models.zip" from https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing, unzip and put them in the folder "nsff_exp/logs/" 

(3) Run the following command for each scene to get quantitative results reported in the paper:
```bash
   # Usage: configs/config_xxx.txt indicates each scene name such as config_balloon1-2.txt in nsff/configs
   python evaluation.py --config configs/config_xxx.txt
```
## Acknowledgment
The code is based on implementation of several prior work:

(1) https://github.com/sniklaus/softmax-splatting

(2) https://github.com/yenchenlin/nerf-pytorch

(3) https://github.com/richzhang/PerceptualSimilarity

(4) https://github.com/intel-isl/MiDaS

(5) https://github.com/princeton-vl/RAFT

(6) https://github.com/NVIDIA/flownet2-pytorch

