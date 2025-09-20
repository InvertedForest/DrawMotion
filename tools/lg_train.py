from collections import defaultdict
import argparse
import copy
import os
import sys
import os.path as osp
workspace_path = osp.abspath(osp.join(__file__, *['..']*2))
print(f'workspace_path: {workspace_path}')
os.chdir(workspace_path)
sys.path.insert(0, workspace_path)
import time

import shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mogen import __version__
from mogen.apis.lg_train import LgModel
from mogen.datasets import build_dataset

from  lightning import Trainer
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from lightning.fabric.fabric import Fabric
from lightning.pytorch import seed_everything
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler
from lightning.pytorch.callbacks import DeviceStatsMonitor
from mmcv.parallel import DataContainer
import my_tools

torch.set_float32_matmul_precision('high')

class UnParaCallback(Callback):
    def __init__(self) -> None:
        pass
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx) -> None:
        for name, param in pl_module.named_parameters():
            if 'clip' in name or 'stick' in name: continue
            if param.grad is None:
                print(name)
                
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('version', help='the version to save logs and models')
    parser.add_argument('gpu', help='-1: 0,1,2,3; -2: 4,5,6,7; else: set to gpu directly')
    args = parser.parse_args()

    return args

def parse_gpu(gpu):
    if gpu == '-1':
        return [0, 1, 2, 3]
    elif gpu == '-2':
        return [4, 5, 6, 7]
    else:
        try:
            return eval(f'[{gpu}]')
        except:
            raise ValueError(f'gpu format error: {gpu}')

def collate_fn(batch):
    keys = batch[0].keys()
    final_batch = {}
    for key in keys:
        if isinstance(batch[0][key], DataContainer):
            # return DataContainer([item.data for item in batch])
            final_batch[key] = [i[key]._data for i in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            # return torch.stack(batch, 0)
            final_batch[key] = torch.stack([item[key] for item in batch], 0)
        else:
            raise NotImplementedError

    return final_batch

def main():
    args = parse_args()
    # seed_everything(123)

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # version is determined in this priority: CLI > segment in file > filename
    if args.version is not None:
        # update configs according to CLI args if args.version is not None
        cfg.version = args.version
    logger = TensorBoardLogger(save_dir='logs', name=cfg.data.train.dataset.dataset_name, version=cfg.version)
    # logger = TensorBoardLogger(save_dir='/root/mnt/occ/virtualdisk/logs', name=cfg.data.train.dataset.dataset_name, version=cfg.version)
    simple_profiler = SimpleProfiler(extended=True)
    advanced_profiler = AdvancedProfiler(filename='profiler.txt')
    # if args.local_rank == 0: 
    if Fabric().global_rank == 0:
        # import my_tools; my_tools.debug()
        workspace = Path.cwd()
        save_dir = Path(logger.log_dir)
        assert workspace.name == 'StickMotion'
        if len([i for i in save_dir.glob('*.ckpt')]):
            raise FileExistsError(f'log already exists')
        if (save_dir / workspace.name).exists():
            shutil.rmtree(save_dir / workspace.name)
        pattern = ['py', 'sh']
        paths = []
        for ext in pattern:
            paths.extend([i for i in workspace.glob(f'**/*.{ext}')])
        for path in paths:
            rel_path = path.relative_to(workspace.parent)
            if rel_path.parts[1] in {'logs', 'data'}:
                continue
            save_path = save_dir / rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, save_path)
        os.symlink(workspace / 'data', save_dir / workspace.name / 'data')
        os.symlink(workspace / 'stickman/weight', save_dir / workspace.name / 'stickman/weight')
        cfg.dump(osp.join(logger.log_dir, osp.basename(args.config)))
        print('workdir copied')



    dataset = build_dataset(cfg.data.train)
    # add an attribute for visualization convenience

    checkpoint_callback = ModelCheckpoint(
            dirpath = logger.log_dir,
            monitor = "all_loss_epoch",
            mode = 'min',
            save_top_k = 10,
            every_n_epochs = 4,
            save_last=True
        )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model = LgModel(cfg)
    para_m = sum(p.numel() for p in model.parameters()) / 1_000_000
    print(f'Parameters: {para_m:.2f}M')
    train_loader = DataLoader(dataset, batch_size=cfg.data.samples_per_gpu, shuffle=True, num_workers=cfg.data.workers_per_gpu, collate_fn=collate_fn)
    trainer = Trainer(accelerator="gpu", 
                      strategy=DDPStrategy(),
                    #   devices=[0],
                    #   devices=[6,7],
                      devices=parse_gpu(args.gpu),
                    #   max_steps=30,
                    #   max_epochs=1,
                      max_epochs=cfg.runner.max_epochs,
                      precision='16-mixed',
                      gradient_clip_algorithm="norm",
                      gradient_clip_val=2,
                      logger=logger,
                      callbacks=[
                          checkpoint_callback,
                          lr_monitor,
                        #   UnParaCallback(),
                        #   DeviceStatsMonitor(cpu_stats=True),   
                          ],
                    #   profiler=advanced_profiler,
                      )     
    # model.load_state_dict(torch.load('/root/StickMotion/logs/kit_ml/tp_loss2/last.ckpt', map_location='cpu')['state_dict']) 
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()

# python tools/lg_train.py configs/remodiffuse/remodiffuse_kit.py  tp_init -1
# po 5 python tools/lg_train.py  configs/remodiffuse/remodiffuse_t2m.py  debug -1