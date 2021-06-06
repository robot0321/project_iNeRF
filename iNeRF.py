import os
import torch
import glob
import cv2
import re

import numpy as np
import imageio
import pprint
from time import time
from tqdm import tqdm, trange
from run_nerf_helpers import *

import matplotlib.pyplot as plt

import run_nerf
import load_blender
from load_blender import load_blender_data

# General setup for GPU device and default tensor type.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

basedir = './logs'
expname = 'blender_paper_lego'

config = os.path.join(basedir, expname, 'config.txt')
print('Args:')
print(open(config, 'r').read())

parser = run_nerf.config_parser()

args = parser.parse_args('--config {}'.format(config))
args.n_gpus = torch.cuda.device_count()

# Create nerf model
render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = run_nerf.create_nerf(args)

bds_dict = {
    'near' : 2.0,
    'far' : 6.0,
}
render_kwargs_train.update(bds_dict)
render_kwargs_test.update(bds_dict)

images, poses, render_poses, hwf, i_split = load_blender_data("./data/nerf_synthetic/lego", False, 8)
H, W, focal = hwf[0], hwf[1], hwf[2]

#################################### setting ##################################
for param in grad_vars:
    param.requires_grad = False
for param in render_kwargs_test['network_fn'].parameters():
    param.requires_grad = False
for param in render_kwargs_test['network_fine'].parameters():
    param.requires_grad = False

# render NeRF network at Target view 
N_rand = args.N_rand
use_batching = False #not args.no_batching
mse = torch.nn.MSELoss()

def get_translated_pose(initpose, learn_trans): # 4x4, 1x6
    w = torch.zeros(3,3).to(learn_trans.device)
    theta = learn_trans[6]
    wnorm = 1 if torch.norm(learn_trans[0:3])==0 else torch.norm(learn_trans[0:3])
    w[0,1] = -learn_trans[2]/wnorm
    w[1,0] = learn_trans[2]/wnorm
    w[0,2] = learn_trans[1]/wnorm
    w[2,0] = -learn_trans[1]/wnorm
    w[1,2] = -learn_trans[0]/wnorm
    w[2,1] = learn_trans[0]/wnorm
    R = torch.matrix_exp(w*theta) # ^c1_c2{R}

    v = learn_trans[3:6].reshape(3,1)
    transform_pose = torch.zeros(4, 4).to(learn_trans.device)
    transform_pose[0:3,0:3] = R
    # transform_pose[0:3,-1] = theta*v.squeeze() + (1-torch.cos(theta))*torch.matmul(w,v).squeeze() + (theta-torch.sin(theta))*torch.matmul(torch.matmul(w,w),v).squeeze()
    transform_pose[0:3,-1] = v.squeeze() + (torch.sin(theta))*torch.matmul(w,v).squeeze() + (1-torch.cos(theta))*torch.matmul(torch.matmul(w,w),v).squeeze()
    transform_pose[3,3] = 1
    newpose = torch.matmul(transform_pose, initpose)

    return newpose

#################################### control variables ##################################
N_iters = 100
step = 100
down = 1
sampledown = 8 # only for video resolution 
sN = 1024*6
renderpath = "./iNeRFrender23"

optlrs = [1e-2, 5e-2, 1e-2] # learn_trl, learn_rot, theta
lrscheduler_gamma = 0.998

init_index = 66
target_index = 8

#################################### iteration ##################################
Targetpose = poses[target_index]
Targetpose = torch.tensor(Targetpose).to(device)
Targetrgb = images[target_index]; Targetrgb[Targetrgb[...,3]==0.]=1.0; Targetrgb = Targetrgb[...,0:3]
Targetrgb = torch.tensor(Targetrgb).to(device)
initrgb = images[init_index]; initrgb[initrgb[...,3]==0.]=1.0; initrgb = initrgb[...,0:3]
initpose = torch.tensor(poses[init_index]).cuda()

# learnable variables
learn_rot = torch.zeros(3).to(device).requires_grad_(True) # w(3), v(3)
learn_trl = torch.zeros(3).to(device).requires_grad_(True) # w(3), v(3)
theta = torch.ones(1).to(device).requires_grad_(True) # w(3), v(3)
learn_trans = torch.cat((learn_rot, learn_trl, theta))

optim = torch.optim.Adam([{'params':learn_trl ,'lr':optlrs[0]}, {'params':learn_rot ,'lr':optlrs[1]}, {'params':theta ,'lr':optlrs[2]}], betas=(0.9, 0.999))
opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, lrscheduler_gamma)

posesave = []
losssave = []
os.makedirs(renderpath, exist_ok=True)

for i in trange(N_iters):
    optim.zero_grad()
    learn_trans = torch.cat((learn_rot, learn_trl, theta))
    newpose = get_translated_pose(initpose, learn_trans)
    if 1:#i==0:
        Hidx, Widx = torch.from_numpy(np.random.choice(np.arange(H//down), size=sN)).to(device), torch.from_numpy(np.random.choice(np.arange(W//down), size=sN)).to(device)
    else:
        with torch.no_grad():
            bestidx = torch.argsort(torch.sum((rgbest-Targetrgb[Hidx, Widx, 0:3])**2, dim=-1), descending=True).to(device)
        newHidx, newWidx = torch.from_numpy(np.random.choice(np.arange(H//down), size=sN-sN//2)).to(device), torch.from_numpy(np.random.choice(np.arange(W//down), size=sN-sN//2)).to(device)
        Hidx, Widx = torch.cat((Hidx[bestidx[:sN//2]], newHidx), dim=0), torch.cat((Widx[bestidx[:sN//2]], newWidx), dim=0)
        # with torch.no_grad():
        #     rgb_sample, _, _, _ = run_nerf.render(H//sampledown, W//sampledown, focal/sampledown, c2w=newpose[:3, :4], **render_kwargs_test)
        #     est_mask = rgb_sample[:,:,0] < 1.0
        #     Hidx, Widx = torch.tensor(np.random.choice(np.arange(H//sampledown), size=sN*5)).to(est_mask.device), torch.tensor(np.random.choice(np.arange(W//sampledown), size=sN*5)).to(est_mask.device)
        #     Hidx, Widx = Hidx[est_mask[Hidx, Widx]][:sN], Widx[est_mask[Hidx, Widx]][:sN]
        #     Hidx, Widx = Hidx*sampledown + torch.randint_like(Hidx, high=3), Widx*sampledown + torch.randint_like(Widx, high=3)

    rgbest, disp, _, _ = run_nerf.render(H//down, W//down, focal/down, c2w=newpose[:3, :4], **render_kwargs_test, chunk=sN, Hidx=Hidx, Widx=Widx)
    loss = mse(rgbest, Targetrgb[Hidx, Widx, 0:3])

    loss.backward()
    optim.step()
    opt_scheduler.step()

    posesave.append(newpose.detach())
    losssave.append(torch.mean((rgbest.detach()-Targetrgb[Hidx, Widx, 0:3])**2))
    if i%(N_iters/step)==0:
        with torch.no_grad():
            rgb_sample, _, _, _ = run_nerf.render(H//sampledown, W//sampledown, focal/sampledown, c2w=newpose[:3, :4], **render_kwargs_test)
            plt.imsave(os.path.join(renderpath, str(i)+".jpg"), np.clip(rgb_sample.detach().cpu().numpy(),  0.0, 1.0))
        
    if i==9: # if not updated
        assert not torch.equal(learn_trans[0:3], torch.zeros(3))
        assert not torch.equal(learn_trans[3:6], torch.zeros(3))

#################################### savepose ##################################
torch.save({"posesave": torch.stack(posesave), 
            "losssave": torch.stack(losssave)
            }, os.path.join(renderpath,"pose_loss_save.pt"))

#################################### video generation ##################################

# renderpath = "./iNeRFrender11_trivial"
imglist = glob.glob(os.path.join(renderpath, "*.jpg"))
convert = lambda text: float(text) if text.isdigit() else text
alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*)', key)]
imglist.sort(key=alphanum)

img = cv2.imread(imglist[0], cv2.IMREAD_COLOR)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps = 10.0

# catimg = np.concatenate((cv2.cvtColor((255.*cv2.resize(initrgb, dsize=(img.shape[0], img.shape[1]), interpolation=cv2.INTER_AREA)).astype('uint8'), cv2.COLOR_RGB2BGR),
#                 img,
#                 cv2.cvtColor((255.*cv2.resize(Targetrgb.detach().cpu().numpy(), dsize=(img.shape[0], img.shape[1]), interpolation=cv2.INTER_AREA)).astype('uint8'), cv2.COLOR_RGB2BGR)), axis=1)

out = cv2.VideoWriter(os.path.join(renderpath, "video.mp4"), fourcc, fps, (3*img.shape[1], img.shape[0]))

for i in range(len(imglist)):
    img = cv2.imread(imglist[i], cv2.IMREAD_COLOR)
    catimg = np.concatenate((cv2.cvtColor((255.*cv2.resize(initrgb, dsize=(img.shape[0], img.shape[1]), interpolation=cv2.INTER_AREA)).astype('uint8'), cv2.COLOR_RGB2BGR),
                img,
                cv2.cvtColor((255.*cv2.resize(Targetrgb.detach().cpu().numpy(), dsize=(img.shape[0], img.shape[1]), interpolation=cv2.INTER_AREA)).astype('uint8'), cv2.COLOR_RGB2BGR)), axis=1)
    out.write(catimg)

out.release()
