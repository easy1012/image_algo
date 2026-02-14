import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from PIL import Image

from utils.common import (
    sample_stratified,
    sample_pdf,
    raw2outputs,
    to8b,
    get_rays
)
from utils.layers import PositionalEncoding, NeRF_layer
from utils.dataset import load_blender


class vanilla_nerf(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoders
        self.enc_x = PositionalEncoding(in_dim=3, num_freqs=10, include_input=True).to(self.device)
        self.enc_d = PositionalEncoding(in_dim=3, num_freqs=4, include_input=True).to(self.device)

        self.data = load_blender(self.opt.data_dir, half_res=self.opt.half_res)

        x_dim = self.enc_x.out_dim()
        d_dim = self.enc_d.out_dim()

        self.model_c = NeRF_layer(x_dim=x_dim, d_dim=d_dim).to(self.device)
        self.model_f = NeRF_layer(x_dim=x_dim, d_dim=d_dim).to(self.device)

    
    def forward(self, data_dir, half_res):
        data = load_blender(data_dir, half_res)
        H, W, focal = data.H, data.W, data.focal
        indices = data.i_val

        idx = int(indices[min(self.opt.index, len(indices) - 1)])
        c2w = data.poses[idx]

        x_dim = self.enc_x.out_dim()
        d_dim = self.enc_d.out_dim()
        rgb = self.render_image(
        model_c=self.model_c,
        model_f=self.model_f,
        enc_x=self.enc_x,
        enc_d=self.enc_d,
        H=H, W=W,
        focal=focal,
        c2w=c2w,
        near=self.opt.near, far=self.opt.far,
        N_samples=self.opt.N_samples,
        N_importance=self.opt.N_importance,
        chunk=self.opt.chunk,
        white_bkgd=self.opt.white_bkgd
    )

        return rgb



    
    @staticmethod
    def _render_rays(
                    model_c: NeRF_layer,
                    model_f: NeRF_layer,
                    enc_x: PositionalEncoding,
                    enc_d: PositionalEncoding,
                    rays_o: torch.Tensor,  # (R,3)
                    rays_d: torch.Tensor,  # (R,3)
                    viewdirs: torch.Tensor,# (R,3)
                    near: float, far: float,
                    N_samples: int,
                    N_importance: int,
                    perturb: bool,
                    white_bkgd: bool):
        # 1) Coarse sampling
        z_vals, pts = sample_stratified(rays_o, rays_d, near, far, N_samples, perturb)
        R, N = z_vals.shape

        pts_flat = pts.reshape(-1, 3)
        vd_flat = viewdirs[:, None, :].expand(R, N, 3).reshape(-1, 3)

        x_enc = enc_x(pts_flat)
        d_enc = enc_d(vd_flat)
        rgb_c, sigma_c = model_c(x_enc, d_enc)
        rgb_c = rgb_c.view(R, N, 3)
        sigma_c = sigma_c.view(R, N, 1)

        rgb_map_c, depth_c, acc_c, weights_c = raw2outputs(rgb_c, sigma_c, z_vals, rays_d, white_bkgd)

        # 2) Fine sampling (hierarchical)
        if N_importance > 0:
            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (R, N-1)
            # Use weights excluding endpoints
            weights_mid = weights_c[:, 1:-1].detach() + 1e-5
            z_samples = sample_pdf(z_vals_mid, weights_mid, N_importance, det=not perturb)
            z_samples = z_samples.detach()

            z_vals_f = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)[0]  # (R, N+Nimp)
            pts_f = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_f[..., None]
            Rf, Nf = z_vals_f.shape

            pts_f_flat = pts_f.reshape(-1, 3)
            vd_f_flat = viewdirs[:, None, :].expand(Rf, Nf, 3).reshape(-1, 3)

            x_enc_f = enc_x(pts_f_flat)
            d_enc_f = enc_d(vd_f_flat)
            rgb_f, sigma_f = model_f(x_enc_f, d_enc_f)
            rgb_f = rgb_f.view(Rf, Nf, 3)
            sigma_f = sigma_f.view(Rf, Nf, 1)

            rgb_map_f, depth_f, acc_f, weights_f = raw2outputs(rgb_f, sigma_f, z_vals_f, rays_d, white_bkgd)
        else:
            rgb_map_f, depth_f, acc_f, weights_f = rgb_map_c, depth_c, acc_c, weights_c

        return rgb_map_f, rgb_map_c

    @torch.no_grad()
    def render_image(self,
        model_c: NeRF_layer,
        model_f: NeRF_layer,
        enc_x: PositionalEncoding,
        enc_d: PositionalEncoding,
        H: int, W: int,
        focal: float,
        c2w: np.ndarray,
        near: float, far: float,
        N_samples: int, N_importance: int,
        chunk: int = 32768,
        white_bkgd: bool = True
    ):
        K = np.array([[focal, 0, 0.5 * W],
                    [0, focal, 0.5 * H],
                    [0, 0, 1]], dtype=np.float32)

        rays_o, rays_d = get_rays(H, W, K, c2w)
        rays_o = torch.from_numpy(rays_o.reshape(-1, 3)).float().cuda()
        rays_d = torch.from_numpy(rays_d.reshape(-1, 3)).float().cuda()
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        out_rgb = []
        for i in range(0, rays_o.shape[0], chunk):
            ro = rays_o[i:i+chunk]
            rd = rays_d[i:i+chunk]
            vd = viewdirs[i:i+chunk]

            rgb = self._render_rays(self.model_c, self.model_f, self.enc_x, self.enc_d, ro, rd, vd,
                            near, far, N_samples, N_importance,
                            perturb=False, white_bkgd=white_bkgd)[0]
            out_rgb.append(rgb.detach().cpu())

        out_rgb = torch.cat(out_rgb, dim=0).view(H, W, 3).numpy()
        return out_rgb

    def train(self):
        H, W, focal = self.data.H, self.data.W, self.data.focal
        images = torch.from_numpy(self.data.images).float().to(self.device)  # (N,H,W,3)
        # poses = torch.from_numpy(self.data.poses).float().to(self.device)  # (N,4,4)
        poses = self.data.poses

        optimizer = torch.optim.Adam(list(self.model_c.parameters()) + list(self.model_f.parameters()), lr=self.opt.lr)

        K = np.array([[focal, 0, 0.5 * W],
                [0, focal, 0.5 * H],
                [0, 0, 1]], dtype=np.float32)

        i_train = self.data.i_train
        print(f"Train views: {len(i_train)} | HxW: {H}x{W} | focal: {focal:.2f}")


        for it in range(1, self.opt.iters + 1):
            # pick a random train image
            img_i = np.random.choice(i_train)
            target = images[img_i]  # (H,W,3)
            c2w = poses[img_i]

            # get rays for this view
            rays_o, rays_d = get_rays(H, W, K, c2w)
            rays_o = torch.from_numpy(rays_o).float().to(self.device)
            rays_d = torch.from_numpy(rays_d).float().to(self.device)

            # random rays
            coords = torch.stack(torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij'), dim=-1)
            coords = coords.reshape(-1, 2)
            select = torch.randint(0, coords.shape[0], (self.opt.batch_rays,), device=self.device)
            sel = coords[select]
            ro = rays_o[sel[:, 0], sel[:, 1]]
            rd = rays_d[sel[:, 0], sel[:, 1]]
            vd = rd / torch.norm(rd, dim=-1, keepdim=True)
            gt = target[sel[:, 0], sel[:, 1]]

            # render
            rgb_f, rgb_c = self._render_rays(
                self.model_c, self.model_f, self.enc_x, self.enc_d,
                ro, rd, vd,
                near=self.opt.near, far=self.opt.far,
                N_samples=self.opt.N_samples, N_importance=self.opt.N_importance,
                perturb=True, white_bkgd=self.opt.white_bkgd
            )

            loss = F.mse_loss(rgb_f, gt)
            if rgb_c is not None:
                loss = loss + 0.1 * F.mse_loss(rgb_c, gt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if it % self.opt.i_print == 0:
                psnr = -10.0 * torch.log10(loss.detach())
                print(f"[{it:06d}/{self.opt.iters}] loss={loss.item():.6f} psnr={psnr.item():.2f}dB")

            if it % self.opt.i_val == 0:
                self.model_c.eval()
                self.model_f.eval()
                # render a validation view (first val)
                val_i = int(self.data.i_val[0])
                val_c2w = poses[val_i]
                rgb_img = self.render_image(
                    self.model_c, self.model_f, self.enc_x, self.enc_d,
                    H, W, focal, val_c2w,
                    near=self.opt.near, far=self.opt.far,
                    N_samples=self.opt.N_samples, N_importance=self.opt.N_importance,
                    white_bkgd=self.opt.white_bkgd
                )
                out = Image.fromarray(to8b(rgb_img))
                out_fp = os.path.join(self.opt.logdir, f"val_{it:06d}.png")
                out.save(out_fp)
                print(f"Saved {out_fp}")
                self.model_c.train()
                self.model_f.train()

            if it % self.opt.i_ckpt == 0:
                ckpt = {
                    "it": it,
                    "model_c": self.model_c.state_dict(),
                    "model_f": self.model_f.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    # 인퍼런스 시 아키텍처/인코딩 파라미터가 동일해야 해서 저장
                    "enc_x_num_freqs": 10,
                    "enc_d_num_freqs": 4,
                    "white_bkgd": self.opt.white_bkgd,
                    "near": self.opt.near,
                    "far": self.opt.far,
                    "N_samples": self.opt.N_samples,
                    "N_importance": self.opt.N_importance,
                    "half_res": self.opt.half_res,
                }
                torch.save(ckpt, self.opt.ckpt_path)
                print(f"Saved checkpoint to {self.opt.ckpt_path}")

    def inference(self, data_dir, half_res):
        rgb = self.forward(data_dir, half_res)
        out_path = os.path.join(self.opt.out_dir, f"render_{self.opt.split}_{self.opt.index:03d}.png")
        Image.fromarray(to8b(rgb)).save(out_path)
        print(f"Saved: {out_path}")

    