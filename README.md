# iNeRF-pytorch

This project is built on [NeRF-pytorch project by Yen-Chen Lin](https://github.com/yenchenlin/nerf-pytorch.git). Please follow the instructions to install the requirements and set the dataset.
This repository is almost same with the code by Yen-Chen Lin except iNeRF.py and iNeRF.ipynb

## How to run?
iNeRF.py and iNeRF.ipynb are basically same. When running the .py file, run
```
CUDA_VISIBLE_DEVICES="0,1" python iNeRF.py
```
---
* Note that there is controll variables in iNeRF.py (see the code). This variables are placed in .ipynb also.
You must set the number of rays (`sN`)  to suit your gpu environment. In my case, 1024x6 rays are used in one Quadro RTX8000 (48GB)
For example, If you use 2 RTX2080Ti (11GB x2), then use under 1024x3 rays

```
N_iters = 100  		# Total iteration
step = 100     		# How many frames in video?
down = 1       		# downsampling for optimal estimation
sampledown = 8 		# only for video resolution 
sN = 1024*6    		# the number of rays 
renderpath = "./iNeRFrender23" # path for saving images, loss, and video

optlrs = [1e-2, 5e-2, 1e-2]    # learn_trl, learn_rot, theta
lrscheduler_gamma = 0.998	# learning rate scheduler

init_index = 66  		# initial pose index
target_index = 8		# target pose index 
```
* The iteration time differs by `sampledown`. If you want iterate fast to see the result quickly, set the `sampledown=16`. The generated video will be low resolution, but it takes only 1~3 minutes. In the case you want to generate high resolution video, set the `sampledown=1`. (takes about 1 hour)

## Experiment
[Notion](https://www.notion.so/iNeRF-6357f04abd21410cb10d7d698fc3e6ac)

## Notice
* Please make your branch and do NOT push on master branch
