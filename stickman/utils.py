import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import interp1d
from tqdm import tqdm
import cv2
from stickman.stickman_data import *
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]
t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

def resample_polyline(polyline, num_samples):
    # Equidistant sampling
    # polyline [n, 2]
    # num_samples int
    distances = np.sqrt(np.sum(np.diff(polyline, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    total_distance = cumulative_distances[-1]
    
    sample_distances = np.linspace(0, total_distance, num_samples)
    
    # Calculate the coordinates of the sampling points using interpolation.
    sample_points = np.zeros((num_samples, 2))
    sample_points[:, 0] = np.interp(sample_distances, cumulative_distances, polyline[:, 0])
    sample_points[:, 1] = np.interp(sample_distances, cumulative_distances, polyline[:, 1])
    
    return sample_points

def interpolate_polygon(vertices, times=1):
    """
    Interpolate the vertices of a polygon
    times - 1 is the number of interpolation points between each pair of vertices
    param vertices: Coordinates of the polygon vertices, shape (n, 2)
    param times: Number of interpolation points for each segment
    return: Interpolated vertex coordinates, shape ((n - 1) * times + 1, 2) 
    """
    assert times > 1 and isinstance(times, int), "times should be an integer and greater than 1."
    
    interpolated_points = np.linspace(vertices[:-1], vertices[1:], times, axis=1, endpoint=False)
    interpolated_vertices = interpolated_points.reshape(-1, vertices.shape[1])
    interpolated_vertices = np.vstack([interpolated_vertices, vertices[-1]])
    
    return interpolated_vertices


def smooth_polygon(vertices, sigma=1.0):
    """
    Smooth the vertices of a polygon
    param vertices: Coordinates of the polygon vertices, shape (n, 2)
    param sigma: Standard deviation of the Gaussian filter
    return: Coordinates of the smoothed vertices 
    """
    x, y = vertices[:, 0], vertices[:, 1]
    x_smooth = gaussian_filter1d(x, sigma=sigma)
    y_smooth = gaussian_filter1d(y, sigma=sigma)
    return np.vstack((x_smooth, y_smooth)).T

# def affine_transform_to_match(a, b, c):
#     '''
#     transform a to b, and apply the same transform to c
#     '''
#     a, b = np.array(a), np.array(b)
#     centroid_a, centroid_b = np.mean(a, axis=0), np.mean(b, axis=0)
#     a_centered, b_centered = a - centroid_a, b - centroid_b
#     H = np.dot(a_centered.T, b_centered)
#     U, S, Vt = np.linalg.svd(H)
#     R = np.dot(Vt.T, U.T)
#     if np.linalg.det(R) < 0: Vt[-1, :] *= -1; R = np.dot(Vt.T, U.T)
#     scale = np.sqrt(np.sum(S) / np.trace(np.dot(a_centered.T, a_centered)))
#     t = centroid_b - scale * np.dot(R, centroid_a)
#     return scale * np.dot(c, R.T) + t


def affine_transform_to_match(a, b, c):
    '''
    Transform a to b using anisotropic scaling affine transformation, and apply the same transform to c
    '''
    a, b = np.array(a), np.array(b)
    centroid_a, centroid_b = np.mean(a, axis=0), np.mean(b, axis=0)
    a_centered = a - centroid_a
    b_centered = b - centroid_b
    
    # Solve for the linear transformation matrix A (allowing anisotropic scaling)
    A, _, _, _ = np.linalg.lstsq(a_centered, b_centered, rcond=None)
    
    # stright to curve have a huge scale
    if abs(A).max() > 3:
        return c
    
    # Calculate the translation vector.
    t = centroid_b - np.dot(centroid_a, A)
    
    # Apply the transformation to c.
    return np.dot(c, A) + t

def draw_joint(joint): #[22, 3]
    import matplotlib.pyplot as plt  
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')  
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'yellow', 'yellow', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    for i, (chain, color) in enumerate(zip(kit_kinematic_chain, colors)):
        if i < 5:
            linewidth = 4.0
        else:
            linewidth = 2.0
        ax.plot3D(joint[chain, 0], joint[chain, 1], joint[chain, 2], linewidth=linewidth,
                    color=color)
    for i, (x, y, z) in enumerate(joint):  
        ax.scatter(x, y, z, s=20, color=[float(i)/255 for i in [30, 30, 30]])  
        ax.text(x, y, z, f'{i}', size=14, zorder=1, color='green')  

    ax.set_xlabel('0')  
    ax.set_ylabel('1')  
    ax.set_zlabel('2') 
    ax.set_aspect('equal', adjustable='box')
    ax.view_init(elev=10, azim=10)
    plt.show()

class StickLocus:
    def __init__(self, dataset_name, smooth=1, shake=1):
        if dataset_name == 'human_ml3d':
            self.limbs_idx = t2m_limbs_idx
        elif dataset_name == 'kit_ml':
            self.limbs_idx = kit_limbs_idx
        else:
            raise NotImplementedError(f'Dataset {dataset_name} not implemented.')
        self._sigma = smooth
        self._shake = shake
        

    def random_param(self):
        # return
        self.shake=np.random.uniform(self._shake/4, self._shake)
        self.sigma=np.random.uniform(self._sigma/10, self._sigma)
        

    
    def norm_locus(self, joint):
        limb_idx = np.array(self.limbs_idx) # [4, 3]
        limb_joint = joint[limb_idx] # [4, 3, 3]
        leg_length = np.linalg.norm(limb_joint[2:, 1:] - limb_joint[2:, :-1], axis=-1).mean()
        locus = joint.copy()[:, 0, [0,2]]
        locus = locus - locus[0] # center
        locus = locus / leg_length # normalize
        return locus
        
    def __call__(self, joint):
        '''
        joint: the joint of stickman [t, 22, 3],
        smooth: the smooth of drawing [0-1],
        shake: the shake of drawing [0-1],
        '''
        self.random_param()
        smooth = self.sigma
        shake = self.shake
        locus = self.norm_locus(joint) # [t, 2]
        # noise
        noise = np.random.uniform(-shake, shake, locus.shape)
        noise = np.cumsum(noise, axis=0)
        noised_locus = locus + noise
        
        # smooth
        x, y = noised_locus[:, 0], noised_locus[:, 1]
        x_smooth = gaussian_filter1d(x, sigma=smooth, mode='nearest')
        y_smooth = gaussian_filter1d(y, sigma=smooth, mode='nearest')
        smoothed_locus = np.vstack((x_smooth, y_smooth)).T
        
        # resample_locus = resample_polyline(smoothed_locus, 64)
        resample_locus = smoothed_locus
        return resample_locus

class StickMan:
    def __init__(self, 
                 dataset_name='kit_ml',
                 smooth:float=4, 
                 shake:float=0.03, 
                 joggle:float=0.01,
                 mismatch:float=0.3,
                 offset:float=0.1,
                 head_deform:float=0.4,
                 stoke_length:float=0.2,
                 part_scale:float=0.1):
        '''
        smooth: the smooth of drawing [0-1],
        shake: the shake of drawing [0-1],
        sharp: the probability of sharp angles occurring at the joints of curved joints when drawing [0-1].
        part_scale: the random scale of each part of the stickman.
        '''
        self.dataset_name = dataset_name
        self._sigma = smooth
        self._shake = shake
        self._joggle = joggle
        self._mismatch = mismatch
        self._offset = offset
        self._part_scale = part_scale
        self._head_deform = head_deform
        self._stoke_length = stoke_length
        if dataset_name == 'human_ml3d':
            key = 't2m'
        elif dataset_name == 'kit_ml':  
            key = 'kit'
        else:
            raise NotImplementedError
        self.spine_idx = eval(f'{key}_spine_idx')
        self.neck_idx = eval(f'{key}_neck_idx')
        self.limbs_idx = eval(f'{key}_limbs_idx')
        self.stand_joint = eval(f'{key}_stand_joint')
        self.joint = None # [22, 3]
        self.tracks = []
        self.idx = 0 # time id: 0 for head, 1 for spine, 2 for right arm, 3 for left arm, 4 for right leg, 5 for left leg
        self.real_scale = 1
        # random.seed(seed)

    def random_param(self):
        # return
        self.shake=np.random.uniform(self._shake/4, self._shake)
        self.joggle=np.random.uniform(self._joggle/4, self._joggle)
        self.sigma=np.random.uniform(self._sigma/10, self._sigma)
        self.offset=np.random.uniform(self._offset/4, self._offset)
        self.mismatch=np.random.uniform(self._mismatch/4, self._mismatch)
        self.part_scale=1 + self._part_scale * np.random.uniform(-1, 1)
        self.head_deform=1 + np.random.uniform(-1, 1)*self._head_deform
        


    def norm_tracks(self):
        assert isinstance(self.tracks, np.ndarray)
        a = self.tracks.copy()
        self.tracks = (a- a.mean(axis=(0,1)))/(a.max(axis=(0,1))-a.min(axis=(0,1))).mean()
        
    def reverse_tracks(self):
        for i,track in enumerate(self.tracks):
            if i > 0: continue # only head
            if np.random.uniform(0, 1) > 0.5:
                self.tracks[i] = track[::-1]
        
    def __call__(self, joint, ori=False, name=None, return_array=False, fig_length=512, point_len=64): #[22, 3]
        '''
        First, a path  is generated, i.e, self.tracks, and then the graph is drawn based on the path.
        '''
        self.joint = joint
        self.normalize()
        self.tracks = []
        # draw head
        self.idx = 0
        self.random_param()
        self.tracks.append(self.draw_head())
        # draw spine
        self.idx = 1
        self.random_param()
        self.tracks.append(self.draw_spine())
        # draw limbs
        for idx in range(len(self.limbs_idx)):
            self.idx = idx + 2
            self.random_param()
            self.tracks.append(self.draw_limb())
        self.combine_tracks()
        self.mismatch_tracks()
        self.get_interpolated_track(point_num=point_len)
        self.joggle_tracks()
        self.norm_tracks()
        self.reverse_tracks()
        if return_array:
            return self.tracks, self.norm_joint
        else:
            self.draw(ori=ori, name=name)

    def joggle_tracks(self):
        for i, track in enumerate(self.tracks):
            self.random_param()
            if np.random.uniform(0, 1) > 0.5:
                noise = np.random.uniform(self.joggle/2, self.joggle, track.shape)
                noise = noise * np.random.choice([-1, 1], track.shape[0])[:, None]
                # noise = np.cumsum(noise, axis=0)
                joggle_track = track + noise
                self.tracks[i] = joggle_track
    
    def combine_tracks(self):
        # random scale
        for i in range(len(self.tracks)):
            self.random_param()
            self.tracks[i] = self.tracks[i] * self.part_scale
        # center
        for i in range(len(self.tracks)):
            self.tracks[i] = self.tracks[i] - self.tracks[i][0]
        # neck
        neck = self.joint[self.neck_idx[0]] - self.joint[self.neck_idx[1]]
        head_track = self.tracks[0]
        # place head track to the origin
        _head_track = head_track - np.mean(head_track, axis=0)
        direct = ((neck[0]*_head_track[:, 0]) > 0) * ((neck[1]*_head_track[:, 1]) > 0)
        _head_track_key = _head_track[:,0]*neck[1] - _head_track[:,1]*neck[0]
        _head_track_key = np.abs(_head_track_key) + (1-direct)*100000
        min_idx = np.argmin(_head_track_key)
        head_combine_point = head_track[min_idx]
        # spine
        self.tracks[1] += head_combine_point + np.random.uniform(-self.offset, self.offset, 2) * neck / np.linalg.norm(neck)
        # arm
        arm_combine_point = self.tracks[1][np.random.choice(len(self.tracks[1])//7)]
        leg_combine_point = self.tracks[1][-1]
        
        for i in range(2, 6):
            if i < 4:
                combine_point = arm_combine_point
            else:
                combine_point = leg_combine_point
            limb_direct = self.joint[self.limbs_idx[i-2][0]] - self.joint[self.limbs_idx[i-2][1]]
            # limb_direct = 2 * ((limb_direct >= 0) - 0.5)
            self.tracks[i] += combine_point - np.random.uniform(self.offset/2, self.offset, 2) * limb_direct / np.linalg.norm(limb_direct)
        # done
        # draw head

    
 
    def mismatch_tracks(self):
        for i in range(len(self.tracks)):
            self.random_param()
            self.tracks[i] += np.random.uniform(-self.mismatch/2, self.mismatch/2, 2)

    
    def noise_line(self, track,  times=1):
        # interpolate
        track = interpolate_polygon(track, times)
        # start brush stoke
        if np.random.uniform(0, 1) > 0.5:
            angle = np.random.uniform(0, 2*np.pi)
            direct = np.array([np.cos(angle), np.sin(angle)])
            direct = direct * np.random.uniform(0, self._stoke_length)
            track[0] = track[1] + direct
        # end brush stoke
        if np.random.uniform(0, 1) > 0.5:
            angle = np.random.uniform(0, 2*np.pi)
            direct = np.array([np.cos(angle), np.sin(angle)])
            direct = direct * np.random.uniform(0, self._stoke_length)
            track[-1] = track[-2] + direct
        # noise
        noise = np.random.uniform(-self.shake, self.shake, track.shape)
        noise = np.cumsum(noise, axis=0)
        noised_track = track + noise
        
        # smooth
        x, y = noised_track[:, 0], noised_track[:, 1]
        x_smooth = gaussian_filter1d(x, sigma=self.sigma, mode='nearest')
        y_smooth = gaussian_filter1d(y, sigma=self.sigma, mode='nearest')
        smoothed_track = np.vstack((x_smooth, y_smooth)).T
        

        return smoothed_track
    
    def noise_align(self, track0):
        times = 20
        track1 = self.noise_line(track0, times=times)
        # length align
        _track_skip = track1[::times]
        track1 = affine_transform_to_match(_track_skip, track0, track1)


        # track2 = self.noise_line(track1,  
        #                         mask_prob=0.5,  
        #                         times=times,
        #                         shake=self.shake,
        #                         sigma=self.sigma*2)
        
        # _track_skip = track2[::times**2]
        # track2 = affine_transform_to_match(_track_skip, track0, track2)
        
        
        return track1

    def draw_head(self):
        # generate a circle
        radius = 1
        down_points = 20
        angles = np.linspace(0, 2 * np.pi, down_points)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        head_track = np.array([x, y]).T # [points, 2]
        # break the circle
        break_point = random.randint(0, len(head_track))
        head_track = np.concatenate([head_track[break_point:], head_track[:break_point]], axis=0)
        # sacle by angle
        angle = np.random.uniform(0, 2*np.pi)
        k = self.head_deform
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        kcs_a = (k-1) * cos_a * sin_a
        c2_a = cos_a ** 2
        trans = np.array([[(k-1)*c2_a+1, kcs_a], # scale head in a direction
                          [kcs_a, k+(1-k)*c2_a]])
        head_track = np.dot(head_track, trans)
        

        head_add = head_track[1] - head_track[0]
        tail_add = head_track[-1] - head_track[-2]
        head_add_point = int(np.random.randint(1, 3))
        tail_add_point = int(np.random.randint(1, 3))
        head_track = np.concatenate([
            [head_track[0]-i*head_add for i in range(head_add_point-1,-1,-1)], 
            head_track, 
            [head_track[-1]+i*tail_add for i in range(1,tail_add_point+1)]
            ], axis=0)

        
        head_track = self.noise_line(head_track, times=6)
        return head_track*0.8
        # extend the head_track
    def draw_spine(self):
        spine_joint = self.joint[self.spine_idx] # [4, 2]
        spine_track = self.noise_align(spine_joint)
        return spine_track

    def random_limb(self, limb_joint, rotate_std=0.18, scale_range=0.12, bend_ratio=0.12):
        root = limb_joint[0].copy()
        out = [root]
        for vector in limb_joint[1:] - limb_joint[:-1]:
            angle = np.random.normal(0, rotate_std)
            scale = 1 + np.random.uniform(-scale_range, scale_range)
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rotated = np.array([
                cos_a * vector[0] - sin_a * vector[1],
                sin_a * vector[0] + cos_a * vector[1]
            ]) * scale
            out.append(out[-1] + rotated)
        out = np.asarray(out)

        direction = out[-1] - out[0]
        norm = max(np.linalg.norm(direction), 1e-6)
        normal = np.array([-direction[1], direction[0]]) / norm
        out[1] += normal * np.random.uniform(-bend_ratio, bend_ratio) * norm
        return out

    
    def draw_limb(self):
        # arm 
        limb_joint = self.joint[self.limbs_idx[self.idx-2]] # [4, 2]
        # arm leg scale
        if self.idx > 3: # leg
            limb_joint = limb_joint * self.arm_leg_scale
            # human bias according to human habit
        human_bias = 0
        limb_joint[1] =  human_bias * limb_joint[0] + (1-human_bias) * limb_joint[1]
        limb_joint = limb_joint[1:]
        # limb_joint = self.random_limb(limb_joint)


        limb_track = self.noise_align(limb_joint)
        return limb_track



    def draw(self, ori=False, name=None):
        if ori:
            ori_tracks = []
            ori_tracks.append(self.joint[self.spine_idx])
            for idx in range(len(self.limbs_idx)):
                ori_tracks.append(self.joint[self.limbs_idx[idx]])
            plt.subplot(121)
            for i, track in enumerate(ori_tracks):
                plt.plot(track[:,0], track[:,1], label=f'{i}', color='blue')
            plt.axis('equal')
            pos = 122
        else:
            pos = 111
            
        plt.subplot(pos)
        for i, track in enumerate(self.tracks):
            plt.plot(track[:,0], track[:,1], label=f'{i}', color='black')
            plt.axis('equal')
        # plt.legend()
        if name is not None:
            plt.title(name)
            plt.savefig(name)
        else:
            plt.show()
        plt.close()

    def cv2_draw(self, length=512):
        self.tracks = self.get_interpolated_track(point_num=64)
        img = np.zeros((length, length), dtype=np.uint8)
        tracks = np.concatenate(self.tracks, axis=0)
        x_length = np.max(tracks[:,0]) - np.min(tracks[:,0])
        y_length = np.max(tracks[:,1]) - np.min(tracks[:,1])
        x_trans = np.min(tracks[:,0])
        y_trans = np.min(tracks[:,1])
        scale = length / max(x_length, y_length)
        self.tracks = [(track-[x_trans, y_trans]) * scale for track in self.tracks]


        for track in self.tracks:
            track[:,1] = length - track[:,1] # opencv y axis is opposite to plt
            points = track.astype(int).reshape((-1, 1, 2))
            cv2.polylines(img, [points], isClosed=False, color=(255), thickness=1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # normalize
        return img.astype(np.float32) / 255

    def get_interpolated_track(self, point_num=100):
        tracks = []
        for track in self.tracks: # [_points, 2] to [point_num, 2]
            interpolated_track = resample_polyline(track, point_num)
            tracks.append(interpolated_track)
       
        tracks = np.array(tracks)
        self.tracks = tracks

    def draw_ori(self):
        self.tracks = []
        self.tracks.append(self.joint[self.spine_idx])
        for idx in range(len(self.limbs_idx)):
            self.tracks.append(self.joint[self.limbs_idx[idx]])
        self.draw()
        


    def normalize(self):
        '''
        Normalize the joint to the range of [0, 1].
        '''
        limb_idx = np.array(self.limbs_idx) # [4, 3]
        limb_joint = self.joint[limb_idx] # [4, 3, 3]
        arm_length = np.linalg.norm(limb_joint[:2, 1:] - limb_joint[:2, :-1], axis=-1).mean() # [4, 3]
        leg_length = np.linalg.norm(limb_joint[2:, 1:] - limb_joint[2:, :-1], axis=-1).mean() # [4, 3]
        self.arm_leg_scale = arm_length / leg_length
        self.real_scale  = (arm_length + leg_length) / 2
        self.joint = self.joint / self.real_scale
        self.joint = self.joint - self.joint[0]
        self.norm_joint = self.joint.copy()
        # rotation x,y to -x, -y, since origin joint faces positive y axis, not compatible with the next calculate, mathplotlib, and blender. To align with the origin joint, the rotated stickman should paired with the origin joint.
        self.joint[:, 0] = -self.joint[:, 0] # y
        self.joint[:, 2] = -self.joint[:, 2] # x
        self.joint = self.joint[:,[0,1]]
        
        
    def jump_stand(self, joint):
        '''
        Jump the stand pose, 5
        '''
        # return 0
        self.joint = joint
        self.normalize()
        # L2 distance
        # distance = (self.joint - self.stand_joint).pow(2).sqrt().sum()
        distance = np.linalg.norm(self.norm_joint - self.stand_joint, axis=-1).sum()
        return distance
    
if __name__ == '__main__':
    import pickle
    import my_tools
    from stickman.eval_with_eye import motion2joint, stick_motion_vis
    import matplotlib.gridspec as gridspec
    
    seed = 7
    np.random.seed(seed)
    random.seed(seed)
    
    title_size = 10
    def track_vis(track, fig, idx, title):
        ax = fig.add_subplot(idx)
        for i in range(6):
            ax.plot(track[i, :, 0], track[i, :, 1], linewidth=1, color='black')
        
        ax.axis('equal')
        ax.axis('off')
        # ax.title.set_text(title)
        ax.title.set_fontsize(title_size)
    def pose_vis(pose, fig, idx, title, joints_num=21):
        if joints_num == 21:
            kinematic_chain = kit_kinematic_chain
        if joints_num == 22:
            kinematic_chain = t2m_kinematic_chain
        ax = fig.add_subplot(idx, projection='3d', aspect='equal')
        colors = ['red', 'blue', 'black', 'red', 'blue',
                'yellow', 'yellow', 'darkblue', 'darkblue', 'darkblue',
                'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

        for i, (chain, color) in enumerate(zip(kinematic_chain, colors)):
            color = 'gray'
            if i < 5:
                linewidth = 2.0
            else:
                linewidth = 1.0
            ax.plot3D(pose[chain, 0], pose[chain, 1], pose[chain, 2], linewidth=linewidth,
                        color=color)
        for i, (x, y, z) in enumerate(pose):  
            ax.scatter(x, y, z, s=20, color=[float(i)/255 for i in [30, 30, 30]])  
            # ax.text(x, y, z, f'{i}', size=14, zorder=1, color='green')  

        # ax.set_xlabel('0')  
        # ax.set_ylabel('1')  
        # ax.set_zlabel('2') 
        ax.set_aspect('equal', adjustable='box')
        ax.view_init(elev=10, azim=10)
        # ax.title.set_text(title)
        ax.title.set_fontsize(0)
        ax.axis('off')

    
    # import torch
    # t2m_motions = np.load('.vscode/t2m_motion_normalized.npy')
    # t2m_motions = torch.tensor(t2m_motions)
    # t2m_joint_num = 22
    # kit_motions = np.load('.vscode/kit_motion_normalized.npy')
    # kit_motions = torch.tensor(kit_motions)
    # kit_joint_num = 21
    
    # column = 5
    # fig = plt.figure(figsize=(20, 10))
    # gs = gridspec.GridSpec(column, 9, width_ratios=[1, 1, 1, 1, 0.1, 1, 1, 1, 1])
    # motion_id = [[[1,0],[100,30]],[[2,0],[210,50]],[[3,0],[230,30]],[[4,0],[240,30]],[[5,0],[150,30]]]
    # for col in range(column):
    #     for i, name in enumerate(['t2m', 'kit']):
    #         motion = eval(name+'_motions')[motion_id[col][i][0], motion_id[col][i][1], None]
    #         joint_num = eval(name+'_joint_num')
    #         joint  = motion2joint(motion, joints_num=joint_num)[0]
    #         pose_vis(joint[:, [2,0,1]], fig, gs[col, 0+5*i], f'title', joint_num)
    #         for j in range(3):
    #             human_stick = StickMan(dataset_name='human_ml3d' if name == 't2m' else 'kit_ml',
    #                                     shake=np.random.uniform(0., 0.01),
    #                                     smooth=np.random.uniform(0.1, 3),
    #                                     mismatch=np.random.uniform(0, 1),
    #                                     direct_scale=0.5,
    #                                     part_scale=0.0)
    #             stick = human_stick(joint, return_array=True)
    #             track_vis(stick[0], fig, gs[col, 1+j+5*i], f'title')
        
        
    # plt.savefig('eval.pdf')
    


