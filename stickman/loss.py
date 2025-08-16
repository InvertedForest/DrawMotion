import torch
from torch import nn
import numpy as np

class SLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()   
        if isinstance(cfg, str):
            dataset_name = cfg
        else:
            dataset_name = cfg.train.dataset_name 
        if dataset_name == 'human_ml3d':
            # self.spine_idx = [12, 9, 3, 0] # spine
            self.spine_idx = [12, 6, 0] # spine
            self.neck_idx = [9,12]
            self.center = 3
            self.limbs_idx = np.array(
                        [[14, 17, 19, 21], # right arm
                        [13, 16, 18, 20], # left arm
                        [0, 2, 5, 8], # right leg
                        [0, 1, 4, 7] # left leg
                        ])
            self.kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
        elif dataset_name == 'kit_ml':  
            self.kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]
            self.spine_idx = [3, 2, 1, 0] # spine
            self.neck_idx = [3,4]
            self.center = 2
            self.limbs_idx = np.array(
                        [[3, 5, 6, 7], # right arm
                        [3, 8, 9, 10], # left arm
                        [0, 11, 12, 13], # right leg
                        [0, 16, 17, 18] # left leg
                        ])
        
        chain = []
        chain_mask = []
        last_i = 0
        for i, sub_chain in enumerate(self.kinematic_chain):
            for joint in sub_chain:
                chain.append(joint)
                if last_i == i:
                    chain_mask.append(1)
                else:
                    chain_mask.append(0)
                last_i = i
        # buffer
        chain_mask.pop(0)
        chain = torch.tensor(chain)
        chain_mask = torch.tensor(chain_mask)

        # limb
        main_limb = torch.tensor(self.limbs_idx[:,1:])
        
        
        self.register_buffer('chain', chain)
        self.register_buffer('chain_mask', chain_mask)
        self.register_buffer('main_limb', main_limb)
                    
                
        
    def forward(self, stickman_feat,  predict_pose, gt_pose):
        '''
        stickman_feat: [b, 128]
        predict_pose: [b, 4, 22, 3]
        gt_pose: [b, 22, 3]
        '''
        ### local        
        predict_offset = predict_pose[:,:,self.chain[1:]] - predict_pose[:,:,self.chain[:-1]] # [b, c, n, 3]
        gt_offset = gt_pose[:,self.chain[1:]] - gt_pose[:,self.chain[:-1]] # [b, n, 3]
        # pred_len = predict_offset.pow(2).sum(-1).sqrt()
        # gt_len = gt_offset.pow(2).sum(-1).sqrt()
        # len_loss = (pred_len-gt_len[:,None]).abs()*self.chain_mask[None,None,...]
        # len_loss = len_loss.mean()
        
        # check: [[i.item(),j.item(),k.item()] for i,j,k in zip(self.chain[1:], self.chain[:-1], self.chain_mask)]
        
        #### candidate loss 1. informaiton loss prevention 2. False information prevention 
        # pow2: the longer stickman stoke shows more precise information
        l2_loss = (predict_offset - gt_offset[:,None]).pow(2)*self.chain_mask[None,None,...,None] # [b, candidate, n, 3]
        candidate_loss = l2_loss.abs().mean([2,3]) # [b, candidate]
        candidate_num = candidate_loss.size(1)

        min_indices = torch.argmin(candidate_loss, dim=1)
        mask = torch.full_like(candidate_loss, 0.1, device=l2_loss.device)
        mask[torch.arange(candidate_loss.size(0)), min_indices] = 1
        # loss = candidate_loss.min(1).values.mean()
        candidate_loss = (candidate_loss*mask).sum(-1).mean()
        
        ### normal loss
        
        # normal_loss = (predict_offset - gt_offset[:,None]).pow(2).mean(dim=tuple(range(predict_offset.dim()-1)))
        # normal_loss = normal_loss[0] + normal_loss[1] + normal_loss[2]*0.1 # x
        
        loss = dict(
            # len_loss=len_loss,
            candidate_loss=candidate_loss
            # normal_loss=normal_loss
        )
        
        return  loss

    def align_forward(self, pred_pose, stick_pose):
        '''
        pred_pose: [b,  22, 3]
        stick_pose: [b, 4, 22, 3]
        '''
        # pred_offset = pred_pose[:,:,self.chain[1:]] - pred_pose[:,:,self.chain[:-1]] # [b, c, n, 3]
        # gt_offset = stick_pose[:,self.chain[1:]] - stick_pose[:,self.chain[:-1]] # [b, n, 3]
        # align_loss = (pred_offset - gt_offset[:,None]).pow(2).mean(dim=tuple(range(pred_offset.dim()-1)))
        pred_pose_length = torch.norm(pred_pose[:,self.main_limb[:,1:]] - pred_pose[:,self.main_limb[:,:-1]], dim=-1).mean(dim=(1,2)) # [b]
        stick_pose_length = torch.norm(stick_pose[:,:,self.main_limb[:,1:]] - stick_pose[:,:,self.main_limb[:,:-1]], dim=-1).mean(dim=(2,3)) # [b,4]
        # aligned_stick_pose = (pred_pose_length[...,None,None,None] / stick_pose_length[...,None,None]) * stick_pose
        # loss = self.forward(None, predict_pose=aligned_stick_pose.clone().detach(), gt_pose=pred_pose)
        aligned_pred_pose = (stick_pose_length.mean(-1)[..., None, None] / pred_pose_length[...,None,None]) * pred_pose
        # loss = self.forward(None, predict_pose=stick_pose.clone().detach(), gt_pose=aligned_pred_pose)
        # return loss['candidate_loss'] * stick_pose.shape[0]
        pred_offset = aligned_pred_pose[:,self.chain[1:]] - aligned_pred_pose[:,self.chain[:-1]] # [b, c, n, 3]
        stick_offset = stick_pose[:,:,self.chain[1:]] - stick_pose[:,:,self.chain[:-1]] # [b, n, 3]
        l2_distance = (pred_offset[:,None] - stick_offset).pow(2)*self.chain_mask[None,None,:,None] # [b, c, n, 3]
        # l2_loss = l2_distance[...,[0,1,2]].abs().mean([2,3])
        l2_loss = l2_distance[...,[0,1,2]].sum(-1).add(1e-4).sqrt().mean(-1)
        # l2_loss = l2_distance[...,[0,2]].abs().mean([2,3])
        loss = l2_loss.min(-1)[0].sum()
        return loss

        
        '''
        from stickman.eval_with_eye import *
        pose = pred_pose[0,None].clone().detach().float().cpu().numpy()
        eval_vis(pose=pose)
        '''