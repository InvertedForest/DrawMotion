import torch as th
import numpy as np
import os
import sys

class WelfordTool:
    def __init__(self, save_mode, ana_step=1_000_000, path='ana_welford.pt', time_steps=None, dims=512):
        """
        完整协方差在线估计(无 M2)
        Complete Covariance Online Estimation (Without M2)
        """
        assert ana_step > 10, 'ana_step must be greater than 10'
        self.ana_step = ana_step
        self.path = path
        self.dims = dims
        self.saved = False
        self.save_mode = save_mode

        if save_mode:
            assert time_steps is not None, 'time_steps is required for analysis record'
            self.steps = th.tensor(time_steps)  # diffusion steps

            S = len(time_steps)
            self.num_samples = th.zeros(S)              # n
            self.step_mu = th.zeros(S, dims)            # μ
            self.step_cov = th.zeros(S, dims, dims)     # Σ
            self.sigma_inv = th.zeros(S, dims, dims)

            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            self.load()

    def to(self, x):
        """Move all tensors to device"""
        if self.save_mode:
            dtype = th.float64
        else:
            dtype = x.dtype
        device = x.device
        for attr in ['step_mu', 'step_cov', 'num_samples', 'steps']:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).to(device=device, dtype=dtype))
        if hasattr(self, 'sigma_inv'):
            self.sigma_inv = self.sigma_inv.to(device=device, dtype=dtype)

    def update(self, mid_query, cur_step):
        """
        mid_query: [B, D]
        cur_step: int
        """
        # Find step index
        mid_query = mid_query.to(dtype=th.float64)
        step_indices = th.where(self.steps == cur_step)[0]
        if len(step_indices) == 0:
            raise ValueError(f'cur_step {cur_step} not in steps {self.steps}')
        idx = step_indices.item()

        n = self.num_samples[idx].item()
        m = mid_query.shape[0]

        if n > self.ana_step and not self.saved:
            self.save()
            self.saved = True
            return

        # Batch mean
        batch_mean = mid_query.mean(0)  # μ_b

        # Batch covariance (unbiased)
        centered = mid_query - batch_mean
        # batch_cov = centered.T @ centered / (m - 1)  # Σ_b
        reg_centered = centered / np.sqrt(m - 1)
        batch_cov = reg_centered.T @ reg_centered

        # First batch
        if n == 0:
            self.step_mu[idx] = batch_mean
            self.step_cov[idx] = batch_cov
            self.num_samples[idx] = m
            return

        # Otherwise incremental update
        mu_old = self.step_mu[idx]
        cov_old = self.step_cov[idx]

        n_new = n + m
        delta = batch_mean - mu_old  # μ_b − μ_old

        mu_new = mu_old + (m / n_new) * delta

        # Σ_new =
        #   (n-1)/(n+m-1) * Σ_old
        # + (m-1)/(n+m-1) * Σ_b
        # + (n*m) / ((n+m)(n+m-1)) * δδ^T
        cov_new = (
            (n - 1) / (n_new - 1) * cov_old
            + (m - 1) / (n_new - 1) * batch_cov
            + (n * m) / (n_new * (n_new - 1)) * th.outer(delta, delta)
        )

        # Save back
        self.step_mu[idx] = mu_new
        self.step_cov[idx] = cov_new
        self.num_samples[idx] = n_new

    @property
    def step_sigma(self):
        """Return covariance directly"""
        return self.step_cov

    def save(self):
        """
        Save analysis results
        """
        if (self.num_samples < 2).any():
            print("Warning: Not enough samples for some steps, skipping save.")
            return

        sigma = self.step_cov
        eps = 1e-6 * th.eye(self.dims, device=sigma.device)
        sigma_stable = sigma + eps.unsqueeze(0)

        try:
            sigma_inv = th.linalg.inv(sigma_stable)

            if th.isnan(sigma_inv).any() or th.isinf(sigma_inv).any():
                print("Warning: Numerical issues in matrix inverse, skipping save.")
                return

            th.save({
                'miu': self.step_mu.cpu(),
                'sigma': sigma.cpu(),
                'sigma_inv': sigma_inv.cpu(),
                'steps': self.steps.cpu(),
                'num_samples': self.num_samples.cpu()
            }, self.path)

        except th.linalg.LinAlgError:
            print("Warning: Matrix inversion failed, skipping save.")

        print(f'Intermediate Feature Statistics saved in {self.path}, stopping the program now. You can run the evaluation command (with multi-gpu) again.')
        os._exit(0)

    def load(self):
        """Load analysis results"""
        data = th.load(self.path)

        self.step_mu = data['miu']
        self.step_cov = data['sigma']
        self.steps = data['steps']
        self.sigma_inv = data['sigma_inv']


        if 'num_samples' in data:
            self.num_samples = data['num_samples']
