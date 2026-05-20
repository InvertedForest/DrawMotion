import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from mmcv import Config

from blender.deal_joint import threed2rot
from mogen.apis.lg_train import LgModel
from stickman.eval_with_eye import motion2joint


def remove_duplicates(polyline, eps=1e-6):
    polyline = np.asarray(polyline, dtype=np.float32)
    diff = np.diff(polyline, axis=0)
    length = np.linalg.norm(diff, axis=1)
    mask = np.concatenate([[True], length > eps])
    return polyline[mask]


def resample_polyline_density(polyline, num_samples, alpha=0.2, eps=1e-6):
    polyline = remove_duplicates(polyline, eps=0.25)
    seg_len = np.linalg.norm(polyline[1:] - polyline[:-1], axis=1)
    max_len = seg_len.max()
    weight = seg_len + max_len * (np.exp(3 * alpha) - 1) / 2
    weight /= weight.sum()
    cdf = np.concatenate([[0], np.cumsum(weight)])
    u = np.linspace(0, 1, num_samples)
    seg_idx = np.searchsorted(cdf, u, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(polyline) - 2)
    seg_u0 = cdf[seg_idx]
    seg_u1 = cdf[seg_idx + 1]
    local_t = (u - seg_u0) / (seg_u1 - seg_u0 + eps)
    return polyline[seg_idx] * (1 - local_t[:, None]) + polyline[seg_idx + 1] * local_t[:, None]


def move_tensors(batch, device):
    output = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def points_from_payload(points):
    return np.asarray([[point["x"], point["y"]] for point in points], dtype=np.float32)


class DrawMotionRunner:
    def __init__(self, ckpt_path, gpu="0", sample_index=0, output_dir="demo/drawmotion_studio/runs"):
        self.root = Path.cwd()
        self.ckpt_path = Path(ckpt_path)
        self.sample_index = int(sample_index)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and str(gpu) != "cpu" else "cpu")
        self.cfg = self.load_config()
        self.dataset_name = self.cfg.dataset_name
        self.joints_num = 21 if self.cfg.input_feats == 251 else 22
        self.model = self.load_model()

    def load_config(self):
        if "kit" in str(self.ckpt_path):
            config_path = "configs/remodiffuse/remodiffuse_kit.py"
        else:
            config_path = "configs/remodiffuse/remodiffuse_t2m.py"
        return Config.fromfile(config_path)

    def load_model(self):
        model = LgModel(self.cfg)
        checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.model.to(self.device)
        model.model.eval()
        model.model.others_cuda()
        return model.model

    def locus_unit_scale(self):
        if self.joints_num == 21:
            return 1000.0
        return 1.0

    def make_empty_batch(self, length, text):
        max_length = int(self.cfg.max_seq_len)
        input_feats = int(self.cfg.input_feats)
        batch = {
            "motion": torch.zeros(1, max_length, input_feats, dtype=torch.float32),
            "motion_mask": torch.zeros(1, max_length, dtype=torch.float32),
            "motion_length": torch.tensor([[length]], dtype=torch.long),
            "clip_feat": None,
            "sample_idx": torch.tensor([self.sample_index], dtype=torch.long),
            "text_idx": torch.tensor([0], dtype=torch.long),
            "stickman_tracks": torch.zeros(1, max_length, 6, 64, 2, dtype=torch.float32),
            "locus": torch.zeros(1, max_length, 2, dtype=torch.float32),
            "stick_mask": torch.zeros(1, max_length, 1, dtype=torch.float32),
            "motion_metas": [{"text": text, "token": None}],
        }
        batch["motion_mask"][0, :length] = 1
        return batch

    def make_batch(self, payload):
        max_length = int(self.cfg.max_seq_len)
        length = int(payload.get("length", max_length))
        length = min(length, max_length)
        batch = self.make_empty_batch(length, payload["text"])

        trajectory = points_from_payload(payload["trajectory"])
        assert len(trajectory) > 1
        traj = resample_polyline_density(trajectory, length, alpha=float(payload.get("density", 0.2)))
        traj = (traj - traj[0, None]) / float(payload.get("trajectory_scale", 50))
        traj[:, 1] *= -1

        model_traj = traj * self.locus_unit_scale()
        batch["locus"][0, :length] = torch.tensor(model_traj, dtype=batch["locus"].dtype)
        batch["locus"][0, length:] = batch["locus"][0, length - 1]

        return move_tensors(batch, self.device), traj

    def prune_runs(self, keep=100):
        run_dirs = sorted(path for path in self.output_dir.iterdir() if path.is_dir())
        for path in run_dirs[:-keep]:
            shutil.rmtree(path)

    def export_result(self, result, traj, run_dir):
        res = result[0]
        motion_length = int(res["motion_length"].item())
        pred_motion = res["pred_motion"][:motion_length]
        unit_scale = self.locus_unit_scale()
        pred_joint = motion2joint(pred_motion, joints_num=self.joints_num) / unit_scale
        gt_motion = res["motion"][:motion_length]
        gt_joint = motion2joint(gt_motion, joints_num=self.joints_num) / unit_scale
        pred_rot = threed2rot(pred_joint)
        gt_traj = res["gt_locus"][:motion_length].cpu().numpy() / unit_scale

        res_dict = {
            "text": res["text"],
            "gt_traj": gt_traj,
            "gt_joint": gt_joint,
            "pred_joint": pred_joint,
            "pred_rot": pred_rot,
        }
        pickle.dump(res_dict, open(run_dir / "all_infor.pkl", "wb"))
        np.save(run_dir / "poly_traj.npy", traj)
        np.savez(run_dir / "draw_input.npz", trajectory=traj)

        payload = {
            "text": res["text"],
            "dataset": self.dataset_name,
            "joints_num": self.joints_num,
            "motion_length": motion_length,
            "input_trajectory": traj.astype(float).tolist(),
            "pred_trajectory": pred_joint[:, 0, [0, 2]].astype(float).tolist(),
            "pred_joint": pred_joint.astype(float).tolist(),
        }
        (run_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def generate(self, payload):
        self.prune_runs()
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "request.json").write_text(json.dumps(payload), encoding="utf-8")
        self.model.guidance.repeat = int(payload["ifg_repeat"])
        self.model.guidance.scale = float(payload["ifg_scale"])
        batch, traj = self.make_batch(payload)
        with torch.inference_mode(False):
            result = self.model(return_loss=False, **batch)
        return self.export_result(result, traj, run_dir)
