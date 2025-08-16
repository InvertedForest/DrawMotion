import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16

from mogen.utils.plot_utils import recover_from_ric
from .base_architecture import BaseArchitecture
from ..builder import (
    ARCHITECTURES,
    build_architecture,
    build_submodule,
    build_loss
)
from ..utils.gaussian_diffusion import (
    GaussianDiffusion, get_named_beta_schedule, create_named_schedule_sampler,
    ModelMeanType, ModelVarType, LossType, space_timesteps, SpacedDiffusion
)
from torch.nn.functional import relu, softmax

def linear_interpolate(tensor, float_index):
    # tensor: [B, T] float_index: [B, N]
    
    int_index = float_index.floor().long() # [B,N]
    frac_index = float_index - int_index.float() # [B,N]

    batch_index = torch.arange(tensor.shape[0], device=tensor.device)[:,None].expand_as(int_index) # [B,N]
    value1 = tensor[batch_index, int_index] # [B, N]
    value2 = tensor[batch_index, int_index + 1] # [B, N]
    
    interpolated_value = value1 * (1 - frac_index) + value2 * frac_index
    
    return interpolated_value # [B, N, M]

def build_diffusion(cfg):
    beta_scheduler = cfg['beta_scheduler']
    diffusion_steps = cfg['diffusion_steps']
    
    betas = get_named_beta_schedule(beta_scheduler, diffusion_steps)
    model_mean_type = {
        'start_x': ModelMeanType.START_X,
        'previous_x': ModelMeanType.PREVIOUS_X,
        'epsilon': ModelMeanType.EPSILON
    }[cfg['model_mean_type']]
    model_var_type = {
        'learned': ModelVarType.LEARNED,
        'fixed_small': ModelVarType.FIXED_SMALL,
        'fixed_large': ModelVarType.FIXED_LARGE,
        'learned_range': ModelVarType.LEARNED_RANGE
    }[cfg['model_var_type']]
    if cfg.get('respace', None) is not None:
        diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(diffusion_steps, cfg['respace']),
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=LossType.MSE
        )
    else:
        diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=LossType.MSE)
    return diffusion


@ARCHITECTURES.register_module()
class MotionDiffusion(BaseArchitecture):

    def __init__(self,
                 model=None,
                 loss_recon=None,
                 loss_weight=None,
                 guidance=None,
                 diffusion_train=None,
                 diffusion_test=None,
                 init_cfg=None,
                 index_num=None,
                 motion_crop=None,
                 inference_type='ddpm',
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.model = build_submodule(model)
        self.loss_recon = build_loss(loss_recon)
        self.diffusion_train = build_diffusion(diffusion_train)
        self.diffusion_test = build_diffusion(diffusion_test)
        self.sampler = create_named_schedule_sampler('uniform', self.diffusion_train)
        self.inference_type = inference_type
        self.loss_weight = loss_weight
        self.guidance = guidance
        self.motion_start = motion_crop[0]
        self.motion_end = motion_crop[1]
        self.index_num = index_num
        
    def others_cuda(self):
        device = [v for k,v in self.model.named_parameters()][0].device
        self.diffusion_train.to(device)
        self.diffusion_test.to(device)
        self.sampler.to(device)


    def forward(self, **kwargs):
        motion, motion_length, motion_mask = kwargs['motion'].float(), kwargs['motion_length'].float(), kwargs['motion_mask'].float()
        # specified_idx, stickman_tracks = kwargs['specified_idx'].int(), kwargs['stickman_tracks'].float()
        stickman_tracks, locus, stick_mask = kwargs['stickman_tracks'].float(), kwargs['locus'].float(), kwargs['stick_mask'].float()
        sample_idx = kwargs.get('sample_idx', None)
        clip_feat = kwargs.get('clip_feat', None)
        B, T = motion.shape[:2]
        text = []
        for i in range(B):
            text.append(kwargs['motion_metas'][i]['text'])

        if self.training:
            t, _ = self.sampler.sample(B, motion.device)
            output = self.diffusion_train.training_losses(
                model=self.model,
                x_start=motion,
                t=t,
                model_kwargs={
                    'motion_mask': motion_mask,
                    'motion_length': kwargs['motion_length'],
                    'locus': locus,
                    'stickman_tracks': stickman_tracks,
                    'text': text,
                    'stick_mask': stick_mask,
                    'clip_feat': clip_feat,
                    'sample_idx': sample_idx}
            )
            all_loss = 0
            pred, target,  p_batch, stick_mask = output['pred'], output['target'],  output['p_batch'], output['stick_mask']
            motion_length = kwargs['motion_length'][:,0]
            loss = {}
            all_loss_batch = self.loss_recon(pred, target, reduction_override='none') # [B, T, M]
            
            loss_item = ['text_loss', 'both_loss', 'stick_loss', 'none_loss']
            assert len(p_batch) == len(loss_item)
            all_batch = sum(p_batch)
            start = 0
            
            joints_num = 21 if motion.shape[-1] == 251 else 22
            for i, batch in enumerate(p_batch):
                if loss_item[i] in {'both_loss', 'stick_loss'}:

                    # stickman
                    mask_s = stick_mask[start:start+batch].squeeze(-1) # [B, T]
                    gt_m = motion[start:start+batch, :, self.motion_start:self.motion_end] # [B, T, M] 
                    pred_m = pred[start:start+batch, :, self.motion_start:self.motion_end]
                    stickman_diff = (gt_m - pred_m).pow(2).mean(-1) # [B, T]
                    stickman_loss = (stickman_diff * mask_s).sum(-1) / (0.1 + mask_s.sum(-1))# [B]
                    stickman_loss = stickman_loss.mean() # [1]
                    loss[f'stickman_{loss_item[i]}'] = stickman_loss
                    
                    # locus
                    pred_joint = recover_from_ric(pred[start:start+batch], joints_num=joints_num, ifnorm=True) # [B, T, J, 3]
                    pred_locus = pred_joint[:, :, 0, [0,2]]/1000 # [B, T, 2]
                    gt_locus = locus[start:start+batch]/1000 # [B, T, 2]
                    locus_diff = (gt_locus - pred_locus).pow(2).mean(-1) # [B, T]
                    mask_l = motion_mask[start:start+batch] # [B, T]
                    locus_loss = (locus_diff * mask_l).sum(-1) / (1 + mask_l.sum(-1)) # [B]
                    locus_loss = locus_loss.mean() # [1]
                    loss[f'locus_{loss_item[i]}'] = locus_loss

                    all_loss = all_loss + (batch / all_batch) * \
                        (stickman_loss * self.loss_weight.stickman_w + \
                         locus_loss * self.loss_weight.locus_w)
                loss[loss_item[i]] = \
                (all_loss_batch[start:start+batch].mean(-1) * \
                motion_mask[start:start+batch]).sum() /\
                motion_mask[start:start+batch].sum()
                all_loss = all_loss + batch/all_batch * loss[loss_item[i]]
                start += batch
            loss['all_loss'] = all_loss
            return loss
        else:
            dim_pose = kwargs['motion'].shape[-1]
            model_kwargs = self.model.get_precompute_condition(device=motion.device,  **kwargs)
            model_kwargs={
                'motion_mask': motion_mask,
                'motion_length': motion_length,
                'locus': locus,
                'stickman_tracks': stickman_tracks,
                'stick_mask': stick_mask,
                'guidance': self.guidance,
                **model_kwargs,
            }
            model_kwargs['stick_joints'] = kwargs.get('stick_joints', None)
            inference_kwargs = kwargs.get('inference_kwargs', {})
            if self.inference_type == 'ddpm':
                output = self.diffusion_test.p_sample_loop(
                    self.model,
                    (B, T, dim_pose),
                    clip_denoised=False,
                    progress=False,
                    model_kwargs=model_kwargs,
                    **inference_kwargs
                )
            else:
                output = self.diffusion_test.ddim_sample_loop(
                    self.model,
                    (B, T, dim_pose),
                    clip_denoised=False,
                    progress=False,
                    model_kwargs=model_kwargs,
                    eta=0,
                    **inference_kwargs
                )
            if getattr(self.model, "post_process") is not None:
                output = self.model.post_process(output)
            results = kwargs
            results['pred_motion'] = output["sample"]
            results = self.split_results(results)
            return results

'''
from stickman.eval_with_eye import *
from blender.deal_joint import threed2rot

[i['text'] for i in results]
[(id, i['text']) for id, i in enumerate(results) if 'jump' in i['text']]

res = results[14]
pred_index = res['pred_index']
specified_idx = res['specified_idx']
pred_motion = res['pred_motion']
gt_motion = res['motion']
mask = res['motion_mask']
stickman = res['stickman_tracks']
start_or_end = 1
pred_id = pred_index.max(0)[1][start_or_end].item(); gt_id = specified_idx[start_or_end].item()
vis_motion = [pred_motion[pred_id, None].double(), gt_motion[gt_id, None].double()]
stick_motion_vis(stickman[start_or_end], vis_motion, res['text'])

# blender
res = results[209]
motion_length = res['motion_length'].item()
# motion = res['pred_motion'][:motion_length]
motion = res['motion'][:motion_length]
np.save('/root/StickMotion/joint.npy', motion2joint(motion, joints_num=22))
scp local_container:joint.npy C:\\Users\\16587\\Desktop\\joint.npy

res = results[67]
motion_length = res['motion_length'].item()
motion = res['pred_motion'][:motion_length]
threed2rot(motion2joint(motion))
# scp local_container:all_infor.pkl C:\\Users\\16587\\Desktop\\all_infor.pkl

res = results[67]
motion_length = res['motion_length'].item()
motion = res['pred_motion'][:motion_length]
threed2rot(motion2joint(motion, joints_num=22))
# scp local_container:all_infor.pkl C:\\Users\\16587\\Desktop\\all_infor.pkl


np.save('/root/StickMotion/joint.npy', joint)
scp local_container:/root/StickMotion/joint.npy C:\\Users\\16587\\Desktop\\joint.npy
'''