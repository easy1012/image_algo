import numpy as np
import torch
from typing import Tuple

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to8b(x: np.ndarray) -> np.ndarray:
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def get_rays(H: int, W: int, K: np.ndarray, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    returns ray origins and directions in world coordinates
    """
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    # pixel to camera
    dirs = np.stack([
        (i - K[0, 2]) / K[0, 0],
        -(j - K[1, 2]) / K[1, 1],
        -np.ones_like(i)
    ], axis=-1)  # (H,W,3)

    # camera to world
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    rays_d = dirs @ R.T
    rays_o = np.broadcast_to(t, rays_d.shape)
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Optional: forward-facing scenes in NDC (original code used for LLFF)
    For Blender synthetic, NDC is usually not needed. Kept for completeness.
    """
    # Shift origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    o0 = -1.0/(W/(2.0*focal)) * rays_o[..., 0]/rays_o[..., 2]
    o1 = -1.0/(H/(2.0*focal)) * rays_o[..., 1]/rays_o[..., 2]
    o2 = 1.0 + 2.0*near/rays_o[..., 2]

    d0 = -1.0/(W/(2.0*focal)) * (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1.0/(H/(2.0*focal)) * (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2.0*near/rays_o[..., 2]

    rays_o = np.stack([o0, o1, o2], axis=-1)
    rays_d = np.stack([d0, d1, d2], axis=-1)
    return rays_o, rays_d


def sample_stratified(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    N_samples: int,
    perturb: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniform sampling in depth, stratified if perturb.
    rays_o, rays_d: (R,3)
    returns:
      z_vals: (R, N_samples)
      pts:    (R, N_samples, 3)
    """
    R = rays_o.shape[0]
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=rays_o.device)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand(R, N_samples)

    if perturb:
        mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape, device=rays_o.device)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]
    return z_vals, pts

def sample_pdf(bins: torch.Tensor, weights: torch.Tensor, N_samples: int, det: bool = False, eps: float = 1e-5):
    """
    Hierarchical sampling (NeRF fine):
    bins: (R, N_bins)  (usually midpoints)
    weights: (R, N_bins)
    returns samples: (R, N_samples)
    """
    weights = weights + eps
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (R, N_bins+1)

    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=bins.device)
        u = u.expand(cdf.shape[0], N_samples)
    else:
        u = torch.rand(cdf.shape[0], N_samples, device=bins.device)

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)

    cdf_below = torch.gather(cdf, 1, below)
    cdf_above = torch.gather(cdf, 1, above)

    bins_below = torch.gather(bins, 1, torch.clamp(below - 1, min=0))  # careful indexing
    bins_above = torch.gather(bins, 1, torch.clamp(above - 1, min=0))

    denom = (cdf_above - cdf_below)
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t = (u - cdf_below) / denom
    samples = bins_below + t * (bins_above - bins_below)
    return samples


def raw2outputs(rgb: torch.Tensor, sigma: torch.Tensor, z_vals: torch.Tensor, rays_d: torch.Tensor, white_bkgd: bool):
    """
    rgb:   (R, N, 3)
    sigma: (R, N, 1)
    z_vals:(R, N)
    rays_d:(R, 3)
    returns: rgb_map (R,3), depth_map (R,), acc_map (R,), weights (R,N)
    """
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[:, :1])], dim=-1)  # (R,N)
    dists = dists * torch.norm(rays_d[:, None, :], dim=-1)  # convert to real distance

    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)  # (R,N)

    # T_i = prod_{j<i} (1-alpha_j)
    T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.0 - alpha + 1e-10], dim=-1), dim=-1)
    T = T[:, :-1]
    weights = alpha * T  # (R,N)

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # (R,3)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights
