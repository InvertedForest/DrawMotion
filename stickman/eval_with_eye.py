import io
import matplotlib.pyplot as plt
import torch
import numpy as np
from mogen.utils.plot_utils import recover_from_ric
from scipy.ndimage import gaussian_filter
from matplotlib.collections import LineCollection
import os

mean_kit = torch.tensor(np.load('data/datasets/kit_ml/mean.npy'))
std_kit = torch.tensor(np.load('data/datasets/kit_ml/std.npy'))
mean_t2m = torch.tensor(np.load('data/datasets/human_ml3d/mean.npy'))
std_t2m = torch.tensor(np.load('data/datasets/human_ml3d/std.npy'))

# [[a[i],a[i+1]] for a in aa for i in range(len(a)-1)]
kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]
t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
title_size=30

def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")  
    return motion.reshape(motion.shape[0], -1, 3)

def track_vis(track, fig, idx, title='track'):
    ax = fig.add_subplot(*idx)
    for i in range(6):
        # ax.plot(track[i, :, 0], track[i, :, 1], linewidth=4, color='black')
        x = track[i, :, 0]
        y = track[i, :, 1]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, 1)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        color = np.linspace(0, 0.5, len(x))
        color[::2] = 0.7
        lc.set_array(color)
        lc.set_linewidth(1)

        ax.add_collection(lc)
    
    ax.axis('equal')
    ax.title.set_text(title)
    ax.title.set_fontsize(title_size)

def track_vis_main(track):
    fig = plt.figure(figsize=(20, 20))
    track_vis(track, fig, 1)
    plt.savefig('track.pdf')

def pose_vis(pose, fig, idx, title, joints_num=21):
    if joints_num == 21:
        kinematic_chain = kit_kinematic_chain
    if joints_num == 22:
        kinematic_chain = t2m_kinematic_chain
    ax = fig.add_subplot(*idx, projection='3d', aspect='equal')
    # ax = fig.add_subplot(411, projection='3d', aspect='equal')
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'yellow', 'yellow', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    for i, (chain, color) in enumerate(zip(kinematic_chain, colors)):
        if i < 5:
            linewidth = 2.0
        else:
            linewidth = 1.0
        ax.plot3D(pose[chain, 0], pose[chain, 1], pose[chain, 2], linewidth=linewidth,
                    color=color)
    for i, (x, y, z) in enumerate(pose):  
        ax.scatter(x, y, z, s=10, color=[float(i)/255 for i in [30, 30, 30]])  
        ax.text(x, y, z, f'{i}', size=7, zorder=1, color='green')  

    ax.set_xlabel('-2')  
    ax.set_ylabel('-0')  
    ax.set_zlabel('1') 
    ax.set_aspect('equal', adjustable='box')
    # ax.view_init(elev=10, azim=10)
    ax.view_init(elev=0, azim=0)
    ax.title.set_text(title)
    ax.title.set_fontsize(title_size)

def feat_heatmap_vis(feat, fig, idx):
    # feat [2, 128]
    ax = fig.add_subplot(idx)
    color = 'viridis'
    ax.imshow(feat, cmap=color)
    ax.axis('auto')
    ax.title.set_text('stick feat & motion feat')
    ax.title.set_fontsize(title_size)

def eval_vis(track=None, pose=None, joints_num=21, title=None):
    '''
    track [batch, 6, 64, 2]
    pose [batch, 21/22, 3]
    '''
    subplot_num = 0
    track_title = []
    pose_title = []
    if track is not None:
        subplot_num += len(track)
        track_title = [f'{i}' for i in range(len(track))]
    if pose is not None:
        subplot_num += len(pose)
        pose_title = [f'{i}' for i in range(len(pose))]
    
    if title is not None:
        assert len(title) == subplot_num
    else:
        title = track_title + pose_title
        
    cols =  4
    rows = (subplot_num + cols - 1) // cols
    fig = plt.figure(figsize=(20, 20))
    subp = 0
    if track is not None:
        for i, t in enumerate(track):
            track_vis(track=t, fig=fig, idx=(rows, cols, subp+1), title=title[subp])
            subp += 1
    if pose is not None:
        for i, p in enumerate(pose):
            # refer to SticMman.Normalize
            p = p[:, [2,0,1]]
            p[:, 0] = -p[:, 0]
            p[:, 1] = -p[:, 1]
            pose_vis(pose=p, fig=fig, idx=(rows, cols, subp+1), title=title[subp], joints_num=joints_num)
            subp += 1
    plt.savefig('eval.pdf')
    save_abs_path = os.getcwd() + '/eval.pdf'
    print(save_abs_path)
    
def pose_vis_online(pose, joints_num=21):
    fig = plt.figure(figsize=(20, 20))
    subp = 1
    for i, p in enumerate(pose):
        # refer to SticMman.Normalize
        p = p[:, [2,0,1]]
        p[:, 0] = -p[:, 0]
        p[:, 1] = -p[:, 1]
        pose_vis(pose=p, fig=fig, idx=(1,len(pose),subp), title=1, joints_num=joints_num)
        subp += 1
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf
    
    

def motion_vis(motion1, motion2):
    joints_num=23
    fig = plt.figure(figsize=(20, 20))
    for id, motion in enumerate([motion1, motion2]):
        joint = recover_from_ric(motion, joints_num).cpu().numpy()[0]
        pose_vis(joint[:, [2,0,1]], fig, id+1, f'{id}')
    plt.savefig('eval.pdf')

def stick_motion_vis(sticks, motions, title, joints_num=21):
    if joints_num == 21:
        mean = mean_kit
        std = std_kit
    if joints_num == 22:
        mean = mean_t2m
        std = std_t2m
    fig = plt.figure(figsize=(20, 20))
    for sid, stick in enumerate(sticks):
        track_vis(stick, fig, 1+sid, title)
        if sid != 0: title = ''
    for mid, motion in enumerate(motions):
        motion = motion * std + mean
        joint = recover_from_ric(motion, joints_num).cpu().numpy()[0]
        pose_vis(joint[:, [2,0,1]], fig, mid+sid+2, f'{mid}', joints_num)
    plt.savefig('eval.pdf')

def norm_motion2joint(motion, joints_num=21):
    if joints_num == 21:
        mean = mean_kit
        std = std_kit
    if joints_num == 22:
        mean = mean_t2m
        std = std_t2m
    motion = motion * std + mean
    joint = recover_from_ric(motion, joints_num).cpu().numpy()
    return joint

def recover_rel_from_ric(data, joints_num):
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    return positions


def motion2joint(motion, joints_num=21, relative=False, smooth=True):
    if joints_num == 21:
        mean = mean_kit
        std =std_kit
    if joints_num == 22:
        mean = mean_t2m
        std =std_t2m
    motion = motion * std + mean
    if relative:
        joint = recover_rel_from_ric(motion, joints_num)
    else:
        joint = recover_from_ric(motion, joints_num)
        if smooth:
            joint = joint.cpu().numpy()
            joint = motion_temporal_filter(joint, sigma=1)
    return joint

