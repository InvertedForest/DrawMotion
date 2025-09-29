import torch
import torch.optim as optim
from mogen.core.optimizer.builder import build_optimizers
from mogen.models import build_architecture
from lightning import LightningModule, LightningDataModule
import pickle
from lightning.fabric.fabric import Fabric
import os
import time
from mogen.utils.plot_utils import recover_from_ric
import numpy as np

class LgModel(LightningModule):

    def __init__(self, cfg, dataset=None, unit=0):
        super().__init__()
        self.save_hyperparameters(cfg._cfg_dict)
        self.cfg = cfg
        self.model = build_architecture(cfg.model)
        self.dataset = dataset
        self.outputs = []
        self.unit = unit 
        self.val_step = -1

    
    def on_train_epoch_start(self) -> None:
        self.model.others_cuda()

        # torch.cuda.empty_cache()
    #     device = self.device
    #     free_memory, total_memory = torch.cuda.mem_get_info(device)
    #     dtype = torch.uint8
    #     element_size = torch.tensor([], dtype=dtype, device=device).element_size()
    #     num_elements = int((free_memory // element_size) * 0.95)
    #     try:
    #         empty_tensor = torch.empty(num_elements, dtype=dtype, device=device)
    #     except RuntimeError as e:
    #         pass
    #     return super().on_train_epoch_start()
    

    def on_validation_start(self) -> None:
        self.model.others_cuda()


    def configure_optimizers(self):
        optimizer = build_optimizers(self.model, self.cfg.optimizer)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs*2)
        milestone = 5/6 * self.trainer.max_epochs
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        loss_dict = self.model.forward(**batch)
        all_loss = loss_dict['all_loss']
        for name, value in loss_dict.items():
            if name == 'all_loss': continue
            self.log(name, value, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('all_loss', all_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return all_loss
    
    def on_validation_epoch_start(self):
        self.stickman_model = StickModel.load_from_checkpoint(self.cfg.stickman_all_path, map_location=self.device, cfg=self.cfg.stick_set, strict=False)
        return super().on_validation_epoch_start()
    
    def validation_step(self, batch):
        # lightning not support max_steps in validation, so we use val_step to control the number of steps
        self.val_step += 1
        if self.trainer.max_steps > 0 and self.val_step > self.trainer.max_steps - 1: return
        # add stickman feature to avoid redundant computation
        
        '''
        #### for stickman training free guidance #####
        track = batch['stickman_tracks']
        B, T, *E = track.shape
        track  = track.reshape(B*T, *E)
        stickman_feat, pred_motion = self.stickman_model(track)
        stickman_feat = stickman_feat.reshape(B, T, *stickman_feat.shape[1:])
        pred_motion = pred_motion.reshape(B, T, *pred_motion.shape[1:])
        # batch['stickman_emb'] = stickman_feat
        pred_motion = (pred_motion[..., 1:, :] - pred_motion[..., 1, None, :]).detach() # [B, T, 4, J-1, 3]
        pred_motion = torch.cat([torch.zeros(B,T,4,1,3, device=pred_motion.device), pred_motion], dim=-2) # [B, T, 4, J, 3]tc
        batch['stick_joints'] = pred_motion
        '''
        

        '''
        ##### for interaction demo #####
        batch['motion_metas'][0]['text'] = 'A person walked casually.'
        batch['motion_metas'][0]['token'] = None
        traj = np.load('.vscode/interaction/poly_traj.npy')
        stickm = np.load('.vscode/interaction/stickman_input.npy')
        leng = traj.shape[0]
        # locus
        batch['locus'][0][:leng] = torch.tensor(traj)
        # motion mask
        batch['motion_mask'][0, :leng] = 1
        batch['motion_mask'][0, leng:] = 0
        batch['motion_length'][0] = leng
        batch['clip_feat'] = None
        # stickman
        stick_index = 70
        batch['stick_mask'][0, ...] = 0
        batch['stick_mask'][0, stick_index, 0] = 1
        batch['stickman_tracks'][0, stick_index] = torch.tensor(stickm)

        '''
        
        # import time; time1 = time.time()
        output = self.model(return_loss=False, **batch)
        # print('model time:', time.time() - time1)
        self.outputs.append(output)

    # def on_validation_end(self) -> None:
    #     # gather the results from all processes
    def on_validation_epoch_end(self) -> None:
        self.outputs = [i for j in self.outputs for i in j]
        tmp_file = f'/dev/shm/{self.unit}_{self.global_rank}.pkl'
        pickle.dump(self.outputs, open(tmp_file, 'wb'))
        self.trainer.strategy.barrier()
        part_list = []
        if self.global_rank == 0:
            for rank in range(self.trainer.num_devices):
                tmp_file = f'/dev/shm/{self.unit}_{rank}.pkl'
                outputs = pickle.load(open(tmp_file, 'rb'))
                os.remove(tmp_file)
                part_list.append(outputs)
            ordered_results = []
            for res in zip(*part_list):
                ordered_results.extend(list(res))
            joints_num = 21 if res[0]['motion'].shape[-1] == 251 else 22
            ordered_results = ordered_results[:len(self.dataset)]
            print(f'StiSim:{1-evalute_sim(ordered_results, joints_num=joints_num)/evalute_mean(ordered_results, joints_num=joints_num)}')
            print(f'LoDist:{evalute_locus(ordered_results, joints_num=joints_num)}')
            print(f'traj error:{evalute_trajectory_error(ordered_results, joints_num=joints_num)}')
            results = self.dataset.evaluate(ordered_results)
            for k, v in results.items():
                print(f'\n{k} : {v:.4f}')


def evalute_trajectory_error(results, joints_num):
    
    dis_list = []
    for result in results:
        length = result['motion_length'].item()
        gt_motion = result['motion'][:length]
        pred_motion = result['pred_motion'][:length]
        gt_joint = recover_from_ric(gt_motion, joints_num=joints_num, ifnorm=True)
        pred_joint = recover_from_ric(pred_motion, joints_num=joints_num, ifnorm=True) 
        scale = 1000 if joints_num == 21 else 1
        gt_locus = gt_joint[:,0,[0,2]]/scale
        pred_locus = pred_joint[:,0,[0,2]]/scale
        dist = (pred_locus - gt_locus).pow(2).sum(-1).sqrt() # [motion_length]
        dist_mean = dist.mean() # [1]
        traj_fail_02 = (dist_mean > 0.2).float()
        traj_fail_05 = (dist_mean > 0.5).float()
        all_fail_02 = (dist > 0.2).float().mean()
        all_fail_05 = (dist > 0.5).float().mean()
        dis_list.append([traj_fail_02.item(), traj_fail_05.item(), all_fail_02.item(), all_fail_05.item(), dist_mean.item()])

    traj_error = torch.tensor(dis_list).mean().item()
    return traj_error

def evalute_locus(results, joints_num):
    
    dis_list = []
    for result in results:
        length = result['motion_length'].item()
        gt_motion = result['motion'][:length]
        pred_motion = result['pred_motion'][:length]
        gt_joint = recover_from_ric(gt_motion, joints_num=joints_num, ifnorm=True)
        pred_joint = recover_from_ric(pred_motion, joints_num=joints_num, ifnorm=True) 
        scale = 1000 if joints_num == 21 else 1
        gt_locus = gt_joint[:,0,[0,2]]/scale
        pred_locus = pred_joint[:,0,[0,2]]/scale
        dist = (pred_locus - gt_locus).pow(2).sum(-1).sqrt().mean()
        dis_list.append(dist.item())
    return sum(dis_list)/len(dis_list)

def evalute_sim(results, joints_num):
    
    sim_list = []
    for result in results:
        length = result['motion_length'].item()
        gt_motion = result['motion'][:,4:3*joints_num+4]
        index = (result['stick_mask'] == 1).nonzero()[:,0]
        gt_stick_motion = gt_motion[index]
        
        pred_motion = result['pred_motion'][:,4:3*joints_num+4]
        pred_stcik_motion = pred_motion[index].double()
        
        dist = (pred_stcik_motion - gt_stick_motion).view(-1,joints_num,3).pow(2).sum(-1).mean()
        
        sim_list.append(dist.item())
    return sum(sim_list)/len(sim_list)


def evalute_mean(results,joints_num):

    sim_list = []
    for result in results:
        length = result['motion_length'].item()
        gt_motion = result['motion'][:,4:3*joints_num+4]
        index = (result['stick_mask'] == 1).nonzero()[:,0]
        gt_stick_motion = gt_motion[index]
        
        pred_motion = result['pred_motion'][:,4:3*joints_num+4]
        dist = (pred_motion[:length,None] - gt_stick_motion[None]).view(-1,joints_num,3).pow(2).sum(-1).mean()
        sim_list.append(dist.item())
    return sum(sim_list)/len(sim_list)

# (evalute_sim(ordered_results, joints_num=21), evalute_mean(ordered_results, joints_num=21))
# 1-evalute_sim(ordered_results, joints_num=21)/evalute_mean(ordered_results, joints_num=21)


from stickman.model import StickmanEncoder, StickmanDecoder

class StickModel(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.stickman_encoder = StickmanEncoder(cfg.stickman_encoder)
        # self.motion_encoder = MotionEncoder(cfg.motion_encoder)
        self.stickman_decoder = StickmanDecoder(cfg.stickman_decoder)

    def forward(self, track):
        """
        track: [B, 6, 64, 2]
        """
        stickman_feat = self.stickman_encoder(track)
        predict_pose = self.stickman_decoder(stickman_feat)
        
        return stickman_feat, predict_pose
