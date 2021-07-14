import os, sys
import numpy as np
import imageio
# import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_nerf_helpers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False
# INFERENCE = True

def splat_rgb_img(ret, ratio, R_w2t, t_w2t, j, H, W, focal, fwd_flow):
    import softsplat

    raw_rgba_s = torch.cat([ret['raw_rgb'], ret['raw_alpha'].unsqueeze(-1)], dim=-1)
    raw_rgba = raw_rgba_s[:, :, j, :].permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
    pts_ref = ret['pts_ref'][:, :, j, :3]

    pts_ref_e_G = NDC2Euclidean(pts_ref, H, W, focal)

    if fwd_flow:
        pts_post = pts_ref + ret['raw_sf_ref2post'][:, :, j, :]
    else:
        pts_post = pts_ref + ret['raw_sf_ref2prev'][:, :, j, :]

    pts_post_e_G = NDC2Euclidean(pts_post, H, W, focal)
    pts_mid_e_G = (pts_post_e_G - pts_ref_e_G) * ratio + pts_ref_e_G

    pts_mid_e_local = se3_transform_points(pts_mid_e_G, 
                                           R_w2t.unsqueeze(0).unsqueeze(0), 
                                           t_w2t.unsqueeze(0).unsqueeze(0))

    pts_2d_mid = perspective_projection(pts_mid_e_local, H, W, focal)

    xx, yy = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    xx = xx.t()
    yy = yy.t()
    pts_2d_original = torch.stack([xx, yy], -1)

    flow_2d = pts_2d_mid - pts_2d_original

    flow_2d = flow_2d.permute(2, 0, 1).unsqueeze(0).contiguous().cuda()

    splat_raw_rgba_dy = softsplat.FunctionSoftsplat(tenInput=raw_rgba, 
                                                 tenFlow=flow_2d, 
                                                 tenMetric=None, 
                                                 strType='average')


    # splatting for static nerf
    pts_rig_e_local = se3_transform_points(pts_ref_e_G, 
                                           R_w2t.unsqueeze(0).unsqueeze(0), 
                                           t_w2t.unsqueeze(0).unsqueeze(0))
    
    pts_2d_rig = perspective_projection(pts_rig_e_local, H, W, focal)

    flow_2d_rig = pts_2d_rig - pts_2d_original

    flow_2d_rig = flow_2d_rig.permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
    raw_rgba_rig = torch.cat([ret['raw_rgb_rigid'], ret['raw_alpha_rigid'].unsqueeze(-1)], dim=-1)
    raw_rgba_rig = raw_rgba_rig[:, :, j, :].permute(2, 0, 1).unsqueeze(0).contiguous().cuda()

    splat_raw_rgba_rig = softsplat.FunctionSoftsplat(tenInput=raw_rgba_rig, 
                                                 tenFlow=flow_2d_rig, 
                                                 tenMetric=None, 
                                                 strType='average')

    splat_alpha_dy = splat_raw_rgba_dy[0, 3:4, :, :]
    splat_rgb_dy = splat_raw_rgba_dy[0, 0:3, :, :]

    splat_alpha_rig = splat_raw_rgba_rig[0, 3:4, :, :]
    splat_rgb_rig = splat_raw_rgba_rig[0, 0:3, :, :]


    return splat_alpha_dy, splat_rgb_dy, splat_alpha_rig, splat_rgb_rig


from poseInterpolator import *

def render_slowmo_bt(disps, render_poses, bt_poses, 
                     hwf, chunk, render_kwargs, 
                     gt_imgs=None, savedir=None, 
                     render_factor=0, target_idx=10):
    # import scipy.io

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    t = time.time()

    count = 0

    save_img_dir = os.path.join(savedir, 'images')
    # save_depth_dir = os.path.join(savedir, 'depths')
    os.makedirs(save_img_dir, exist_ok=True)
    # os.makedirs(save_depth_dir, exist_ok=True)

    for i, cur_time in enumerate(np.linspace(target_idx - 10., target_idx + 10., 200 + 1).tolist()):
        flow_time = int(np.floor(cur_time))
        ratio = cur_time - np.floor(cur_time)
        print('cur_time ', i, cur_time, ratio)
        t = time.time()

        int_rot, int_trans = linear_pose_interp(render_poses[flow_time, :3, 3], 
                                                render_poses[flow_time, :3, :3],
                                                render_poses[flow_time + 1, :3, 3], 
                                                render_poses[flow_time + 1, :3, :3], 
                                                ratio)

        int_poses = np.concatenate((int_rot, int_trans[:, np.newaxis]), 1)
        int_poses = np.concatenate([int_poses[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

        int_poses = np.dot(int_poses, bt_poses[i])

        render_pose = torch.Tensor(int_poses).to(device)

        R_w2t = render_pose[:3, :3].transpose(0, 1)
        t_w2t = -torch.matmul(R_w2t, render_pose[:3, 3:4])

        num_img = gt_imgs.shape[0]
        img_idx_embed_1 = (np.floor(cur_time))/float(num_img) * 2. - 1.0
        img_idx_embed_2 = (np.floor(cur_time) + 1)/float(num_img) * 2. - 1.0

        print('img_idx_embed_1 ', cur_time, img_idx_embed_1)

        ret1 = render_sm(img_idx_embed_1, 0, False,
                        num_img, 
                        H, W, focal, 
                        chunk=1024*16, 
                        c2w=render_pose,
                        **render_kwargs)

        ret2 = render_sm(img_idx_embed_2, 0, False,
                        num_img, 
                        H, W, focal, 
                        chunk=1024*16, 
                        c2w=render_pose, 
                        **render_kwargs)

        T_i = torch.ones((1, H, W))
        final_rgb = torch.zeros((3, H, W))
        num_sample = ret1['raw_rgb'].shape[2]
        # final_depth = torch.zeros((1, H, W))
        z_vals = ret1['z_vals']

        for j in range(0, num_sample):
            splat_alpha_dy_1, splat_rgb_dy_1, \
            splat_alpha_rig_1, splat_rgb_rig_1 = splat_rgb_img(ret1, ratio, R_w2t, t_w2t, 
                                                            j, H, W, focal, True)
            splat_alpha_dy_2, splat_rgb_dy_2, \
            splat_alpha_rig_2, splat_rgb_rig_2 = splat_rgb_img(ret2, 1. - ratio, R_w2t, t_w2t, 
                                                            j, H, W, focal, False)

            final_rgb += T_i * (splat_alpha_dy_1 * splat_rgb_dy_1 + \
                                splat_alpha_rig_1 * splat_rgb_rig_1 ) * (1.0 - ratio)
            final_rgb += T_i * (splat_alpha_dy_2 * splat_rgb_dy_2 + \
                                splat_alpha_rig_2 * splat_rgb_rig_2 ) * ratio
            # splat_alpha = splat_alpha1 * (1. - ratio) + splat_alpha2 * ratio
            # final_rgb += T_i * (splat_alpha1 * (1. - ratio) * splat_rgb1 +  splat_alpha2 * ratio * splat_rgb2)

            alpha_1_final = (1.0 - (1. - splat_alpha_dy_1) * (1. - splat_alpha_rig_1) ) * (1. - ratio)
            alpha_2_fianl = (1.0 - (1. - splat_alpha_dy_2) * (1. - splat_alpha_rig_2) ) * ratio
            alpha_final = alpha_1_final + alpha_2_fianl

            # final_depth += T_i * (alpha_final) * z_vals[..., j]
            T_i = T_i * (1.0 - alpha_final + 1e-10)

        filename = os.path.join(savedir, 'slow-mo_%03d.jpg'%(i))
        rgb8 = to8b(final_rgb.permute(1, 2, 0).cpu().numpy())

        # final_depth = torch.clamp(final_depth/percentile(final_depth, 98), 0., 1.) 
        # depth8 = to8b(final_depth.permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy())

        start_y = (rgb8.shape[1] - 512) // 2
        rgb8 = rgb8[:, start_y:start_y+ 512, :]
        # depth8 = depth8[:, start_y:start_y+ 512, :]

        filename = os.path.join(save_img_dir, '{:03d}.jpg'.format(i))
        imageio.imwrite(filename, rgb8)

        # filename = os.path.join(save_depth_dir, '{:03d}.jpg'.format(i))
        # imageio.imwrite(filename, depth8)

def render_lockcam_slowmo(ref_c2w, num_img, 
                        hwf, chunk, render_kwargs, 
                        gt_imgs=None, savedir=None, 
                        render_factor=0,
                        target_idx=5):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    t = time.time()

    count = 0

    for i, cur_time in enumerate(np.linspace(target_idx - 8., target_idx + 8., 160 + 1).tolist()):
        ratio = cur_time - np.floor(cur_time)

        render_pose = ref_c2w[:3,:4] #render_poses[i % num_frame_per_cycle][:3,:4]

        R_w2t = render_pose[:3, :3].transpose(0, 1)
        t_w2t = -torch.matmul(R_w2t, render_pose[:3, 3:4])

        num_img = gt_imgs.shape[0]
        img_idx_embed_1 = (np.floor(cur_time))/float(num_img) * 2. - 1.0
        img_idx_embed_2 = (np.floor(cur_time) + 1)/float(num_img) * 2. - 1.0
        print('render lock camera time ', i, cur_time, ratio, time.time() - t)
        t = time.time()

        ret1 = render_sm(img_idx_embed_1, 0, False,
                        num_img, 
                        H, W, focal, 
                        chunk=1024*16, 
                        c2w=render_pose,
                        **render_kwargs)

        ret2 = render_sm(img_idx_embed_2, 0, False,
                        num_img, 
                        H, W, focal, 
                        chunk=1024*16, 
                        c2w=render_pose, 
                        **render_kwargs)

        T_i = torch.ones((1, H, W))
        final_rgb = torch.zeros((3, H, W))
        num_sample = ret1['raw_rgb'].shape[2]

        for j in range(0, num_sample):
            splat_alpha_dy_1, splat_rgb_dy_1, \
            splat_alpha_rig_1, splat_rgb_rig_1 = splat_rgb_img(ret1, ratio, R_w2t, 
                                                               t_w2t, j, H, W, 
                                                               focal, True)
            splat_alpha_dy_2, splat_rgb_dy_2, \
            splat_alpha_rig_2, splat_rgb_rig_2 = splat_rgb_img(ret2, 1. - ratio, R_w2t, 
                                                               t_w2t, j, H, W, 
                                                               focal, False)

            final_rgb += T_i * (splat_alpha_dy_1 * splat_rgb_dy_1 + \
                                splat_alpha_rig_1 * splat_rgb_rig_1 ) * (1.0 - ratio)
            final_rgb += T_i * (splat_alpha_dy_2 * splat_rgb_dy_2 + \
                                splat_alpha_rig_2 * splat_rgb_rig_2 ) * ratio

            alpha_1_final = (1.0 - (1. - splat_alpha_dy_1) * (1. - splat_alpha_rig_1) ) * (1. - ratio)
            alpha_2_fianl = (1.0 - (1. - splat_alpha_dy_2) * (1. - splat_alpha_rig_2) ) * ratio
            alpha_final = alpha_1_final + alpha_2_fianl

            T_i = T_i * (1.0 - alpha_final + 1e-10)

        filename = os.path.join(savedir, '%03d.jpg'%(i))
        rgb8 = to8b(final_rgb.permute(1, 2, 0).cpu().numpy())

        start_y = (rgb8.shape[1] - 512) // 2
        rgb8 = rgb8[:, start_y:start_y+ 512, :]

        imageio.imwrite(filename, rgb8)


def render_sm(img_idx, chain_bwd, chain_5frames,
               num_img, H, W, focal,     
               chunk=1024*16, rays=None, c2w=None, ndc=True,
               near=0., far=1.,
               use_viewdirs=False, c2w_staticcam=None,
               **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays_sm(img_idx, chain_bwd, chain_5frames, 
                               num_img, rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    return all_ret

def batchify_rays_sm(img_idx, chain_bwd, chain_5frames, 
                    num_img, rays_flat, chunk=1024*16, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """

    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays_sm(img_idx, chain_bwd, chain_5frames, 
                            num_img, rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    
    return all_ret


def raw2rgba_blend_slowmo(raw, raw_blend_w, z_vals, rays_d, raw_noise_std=0):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.

    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

    alpha = raw2alpha(raw[...,3] + noise, dists) * raw_blend_w  # [N_rays, N_samples]

    return rgb, alpha



def render_rays_sm(img_idx, 
                chain_bwd,
                chain_5frames,
                num_img,
                ray_batch,
                network_fn,
                network_query_fn, 
                rigid_network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_rigid=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                inference=True):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    img_idx_rep = torch.ones_like(pts[:, :, 0:1]) * img_idx
    pts_ref = torch.cat([pts, img_idx_rep], -1)

    # query point at time t
    rgb_map_rig, depth_map_rig, raw_rgba_rigid, raw_blend_w = get_rigid_outputs(pts_ref, viewdirs, 
                                                                               rigid_network_query_fn, 
                                                                               network_rigid, 
                                                                               z_vals, rays_d, 
                                                                               raw_noise_std)

    # query point at time t
    raw_ref = network_query_fn(pts_ref, viewdirs, network_fn)
    raw_rgba_ref = raw_ref[:, :, :4]
    raw_sf_ref2prev = raw_ref[:, :, 4:7]
    raw_sf_ref2post = raw_ref[:, :, 7:10]
    # raw_blend_w_ref = raw_ref[:, :, 12]

    raw_rgb, raw_alpha = raw2rgba_blend_slowmo(raw_rgba_ref, raw_blend_w, 
                                            z_vals, rays_d, raw_noise_std)
    raw_rgb_rigid, raw_alpha_rigid = raw2rgba_blend_slowmo(raw_rgba_rigid, (1. - raw_blend_w), 
                                                            z_vals, rays_d, raw_noise_std)

    ret = {'raw_rgb': raw_rgb, 'raw_alpha': raw_alpha,  
            'raw_rgb_rigid':raw_rgb_rigid, 'raw_alpha_rigid':raw_alpha_rigid,
            'raw_sf_ref2prev': raw_sf_ref2prev, 
            'raw_sf_ref2post': raw_sf_ref2post,
            'pts_ref':pts_ref, 'z_vals':z_vals}

    return ret


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*16):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs[:, :, :3].shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, 
                            list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(img_idx, chain_bwd, chain_5frames, 
                num_img, rays_flat, chunk=1024*16, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """

    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(img_idx, chain_bwd, chain_5frames, 
                        num_img, rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    
    return all_ret


def render(img_idx, chain_bwd, chain_5frames,
           num_img, H, W, focal,     
           chunk=1024*16, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(img_idx, chain_bwd, chain_5frames, 
                        num_img, rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # k_extract = ['rgb_map', 'disp_map', 'depth_map', 'scene_flow', 'raw_sf_t']

    # ret_list = [all_ret[k] for k in k_extract]
    # ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    # return ret_list + [ret_dict]
    return all_ret


def render_bullet_time(render_poses, img_idx_embed, num_img, 
                    hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()

    save_img_dir = os.path.join(savedir, 'images')
    # save_depth_dir = os.path.join(savedir, 'depths')
    os.makedirs(save_img_dir, exist_ok=True)
    # os.makedirs(save_depth_dir, exist_ok=True)

    for i in range(0, (render_poses.shape[0])):
        c2w = render_poses[i]
        print(i, time.time() - t)
        t = time.time()

        ret = render(img_idx_embed, 0, False,
                     num_img, 
                     H, W, focal, 
                     chunk=1024*32, c2w=c2w[:3,:4], 
                     **render_kwargs)

        depth = torch.clamp(ret['depth_map_ref']/percentile(ret['depth_map_ref'], 97), 0., 1.)  #1./disp
        rgb = ret['rgb_map_ref'].cpu().numpy()#.append(ret['rgb_map_ref'].cpu().numpy())

        if savedir is not None:
            rgb8 = to8b(rgb)
            depth8 = to8b(depth.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy())

            start_y = (rgb8.shape[1] - 512) // 2
            rgb8 = rgb8[:, start_y:start_y+ 512, :]

            # depth8 = depth8[:, start_y:start_y+ 512, :]

            filename = os.path.join(save_img_dir, '{:03d}.jpg'.format(i))
            imageio.imwrite(filename, rgb8)

            # filename = os.path.join(save_depth_dir, '{:03d}.jpg'.format(i))
            # imageio.imwrite(filename, depth8)

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    # XYZ + T
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, 4)

    input_ch_views = 0
    embeddirs_fn = None

    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed, 3)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

    # print(torch.cuda.device_count())
    # sys.exit()

    device_ids = list(range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    grad_vars = list(model.parameters())

    embed_fn_rigid, input_rigid_ch = get_embedder(args.multires, args.i_embed, 3)
    model_rigid = Rigid_NeRF(D=args.netdepth, W=args.netwidth,
                             input_ch=input_rigid_ch, output_ch=output_ch, skips=skips,
                             input_ch_views=input_ch_views, 
                             use_viewdirs=args.use_viewdirs).to(device)

    model_rigid = torch.nn.DataParallel(model_rigid, device_ids=device_ids)

    model_fine = None
    grad_vars += list(model_rigid.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                         embed_fn=embed_fn,
                                                                         embeddirs_fn=embeddirs_fn,
                                                                         netchunk=args.netchunk)

    rigid_network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                               embed_fn=embed_fn_rigid,
                                                                               embeddirs_fn=embeddirs_fn,
                                                                               netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################
    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]

        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step'] + 1
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        model.load_state_dict(ckpt['network_fn_state_dict'])
        # Load model
        print('LOADING SF MODEL!!!!!!!!!!!!!!!!!!!')
        model_rigid.load_state_dict(ckpt['network_rigid'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'rigid_network_query_fn':rigid_network_query_fn,
        'network_rigid' : model_rigid,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'inference': False
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['inference'] = True

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs_blending(raw_dy, 
                         raw_rigid,
                         raw_blend_w,
                         z_vals, rays_d, 
                         raw_noise_std):
    act_fn = F.relu

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb_dy = torch.sigmoid(raw_dy[..., :3])  # [N_rays, N_samples, 3]
    rgb_rigid = torch.sigmoid(raw_rigid[..., :3])  # [N_rays, N_samples, 3]

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_dy[...,3].shape) * raw_noise_std

    opacity_dy = act_fn(raw_dy[..., 3] + noise)#.detach() #* raw_blend_w
    opacity_rigid = act_fn(raw_rigid[..., 3] + noise)#.detach() #* (1. - raw_blend_w) 

    # alpha with blending weights
    alpha_dy = (1. - torch.exp(-opacity_dy * dists) ) * raw_blend_w
    alpha_rig = (1. - torch.exp(-opacity_rigid * dists)) * (1. - raw_blend_w)

    Ts = torch.cumprod(torch.cat([torch.ones((alpha_dy.shape[0], 1)), 
                                (1. - alpha_dy) * (1. - alpha_rig)  + 1e-10], -1), -1)[:, :-1]
    
    weights_dy = Ts * alpha_dy
    weights_rig = Ts * alpha_rig

    # union map 
    rgb_map = torch.sum(weights_dy[..., None] * rgb_dy + \
                        weights_rig[..., None] * rgb_rigid, -2) 

    weights_mix = weights_dy + weights_rig
    depth_map = torch.sum(weights_mix * z_vals, -1)

    # compute dynamic depth only
    alpha_fg = 1. - torch.exp(-opacity_dy * dists)
    weights_fg = alpha_fg * torch.cumprod(torch.cat([torch.ones((alpha_fg.shape[0], 1)), 
                                                                1.-alpha_fg + 1e-10], -1), -1)[:, :-1]
    depth_map_fg = torch.sum(weights_fg * z_vals, -1)
    rgb_map_fg = torch.sum(weights_fg[..., None] * rgb_dy, -2) 

    return rgb_map, depth_map, \
           rgb_map_fg, depth_map_fg, weights_fg, \
           weights_dy


def raw2outputs_warp(raw_p, 
                     z_vals, rays_d, 
                     raw_noise_std=0):

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw_p[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.

    if raw_noise_std > 0.:
        noise = torch.randn(raw_p[...,3].shape) * raw_noise_std

    act_fn = F.relu
    opacity = act_fn(raw_p[..., 3] + noise)

    alpha = 1. - torch.exp(-opacity * dists)

    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)

    return rgb_map, depth_map, weights#, alpha #alpha#, 1. - probs


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.

    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))


    return rgb_map, weights, depth_map

def get_rigid_outputs(pts, viewdirs, 
                      network_query_fn, 
                      network_rigid, 
                      # netowrk_blend,
                      z_vals, rays_d, 
                      raw_noise_std):

    # with torch.no_grad():        
    raw_rigid = network_query_fn(pts[..., :3], viewdirs, network_rigid)
    raw_rgba_rigid = raw_rigid[..., :4]
    raw_blend_w = raw_rigid[..., 4:]

    rgb_map_rig, weights_rig, depth_map_rig = raw2outputs(raw_rgba_rigid, z_vals, rays_d, 
                                                          raw_noise_std, 
                                                          white_bkgd=False)

    return rgb_map_rig, depth_map_rig, raw_rgba_rigid, raw_blend_w[..., 0]


def compute_2d_prob(weights_p_mix, 
                    raw_prob_ref2p):
    prob_map_p = torch.sum(weights_p_mix.detach() * (1.0 - raw_prob_ref2p), -1)
    return prob_map_p

def render_rays(img_idx, 
                chain_bwd,
                chain_5frames,
                num_img,
                ray_batch,
                network_fn,
                network_query_fn, 
                rigid_network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_rigid=None,
                # netowrk_blend=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                inference=False):

    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    img_idx_rep = torch.ones_like(pts[:, :, 0:1]) * img_idx
    pts_ref = torch.cat([pts, img_idx_rep], -1)

    # query point at time t
    rgb_map_rig, depth_map_rig, raw_rgba_rigid, raw_blend_w = get_rigid_outputs(pts_ref, viewdirs, 
                                                                               rigid_network_query_fn, 
                                                                               network_rigid, 
                                                                               z_vals, rays_d, 
                                                                               raw_noise_std)


    raw_ref = network_query_fn(pts_ref, viewdirs, network_fn)
    raw_rgba_ref = raw_ref[:, :, :4]
    raw_sf_ref2prev = raw_ref[:, :, 4:7]
    raw_sf_ref2post = raw_ref[:, :, 7:10]
    # raw_blend_w_ref = raw_ref[:, :, 12]

    rgb_map_ref, depth_map_ref, \
    rgb_map_ref_dy, depth_map_ref_dy, weights_ref_dy, \
    weights_ref_dd = raw2outputs_blending(raw_rgba_ref, raw_rgba_rigid,
                                          raw_blend_w,
                                          z_vals, rays_d, 
                                          raw_noise_std)

    weights_map_dd = torch.sum(weights_ref_dd, -1).detach()

    ret = {'rgb_map_ref': rgb_map_ref, 'depth_map_ref' : depth_map_ref,  
            'rgb_map_rig':rgb_map_rig, 'depth_map_rig':depth_map_rig, 
            'rgb_map_ref_dy':rgb_map_ref_dy, 
            'depth_map_ref_dy':depth_map_ref_dy, 
            'weights_map_dd': weights_map_dd}

    if inference:
        return ret
    else:
        ret['raw_sf_ref2prev'] = raw_sf_ref2prev
        ret['raw_sf_ref2post'] = raw_sf_ref2post
        ret['raw_pts_ref'] = pts_ref[:, :, :3]
        ret['weights_ref_dy'] = weights_ref_dy
        ret['raw_blend_w'] = raw_blend_w

    img_idx_rep_post = torch.ones_like(pts[:, :, 0:1]) * (img_idx + 1./num_img * 2. )
    pts_post = torch.cat([(pts_ref[:, :, :3] + raw_sf_ref2post), img_idx_rep_post] , -1)

    img_idx_rep_prev = torch.ones_like(pts[:, :, 0:1]) * (img_idx - 1./num_img * 2. )    
    pts_prev = torch.cat([(pts_ref[:, :, :3] + raw_sf_ref2prev), img_idx_rep_prev] , -1)

    # render points at t - 1
    raw_prev = network_query_fn(pts_prev, viewdirs, network_fn)
    raw_rgba_prev = raw_prev[:, :, :4]
    raw_sf_prev2prevprev = raw_prev[:, :, 4:7]
    raw_sf_prev2ref = raw_prev[:, :, 7:10]

    # render from t - 1
    rgb_map_prev_dy, _, weights_prev_dy = raw2outputs_warp(raw_rgba_prev,
                                                           z_vals, rays_d, 
                                                           raw_noise_std)


    ret['raw_sf_prev2ref'] = raw_sf_prev2ref
    ret['rgb_map_prev_dy'] = rgb_map_prev_dy
    
    # render points at t + 1
    raw_post = network_query_fn(pts_post, viewdirs, network_fn)
    raw_rgba_post = raw_post[:, :, :4]
    raw_sf_post2ref = raw_post[:, :, 4:7]
    raw_sf_post2postpost = raw_post[:, :, 7:10]

    rgb_map_post_dy, _, weights_post_dy = raw2outputs_warp(raw_rgba_post,
                                                           z_vals, rays_d, 
                                                           raw_noise_std)

    ret['raw_sf_post2ref'] = raw_sf_post2ref
    ret['rgb_map_post_dy'] = rgb_map_post_dy

    raw_prob_ref2prev = raw_ref[:, :, 10]
    raw_prob_ref2post = raw_ref[:, :, 11]

    prob_map_prev = compute_2d_prob(weights_prev_dy,
                                    raw_prob_ref2prev)
    prob_map_post = compute_2d_prob(weights_post_dy, 
                                    raw_prob_ref2post)

    ret['prob_map_prev'] = prob_map_prev
    ret['prob_map_post'] = prob_map_post

    ret['raw_prob_ref2prev'] = raw_prob_ref2prev
    ret['raw_prob_ref2post'] = raw_prob_ref2post

    ret['raw_pts_post'] = pts_post[:, :, :3]
    ret['raw_pts_prev'] = pts_prev[:, :, :3]

    # # ======================================  two-frames chain loss ===============================
    if chain_bwd:
        # render point frames at t - 2
        img_idx_rep_prevprev = torch.ones_like(pts[:, :, 0:1]) * (img_idx - 2./num_img * 2. )    
        pts_prevprev = torch.cat([(pts_prev[:, :, :3] + raw_sf_prev2prevprev), img_idx_rep_prevprev] , -1)
        ret['raw_pts_pp'] = pts_prevprev[:, :, :3]

        if chain_5frames:
            raw_prevprev = network_query_fn(pts_prevprev, viewdirs, network_fn)
            raw_rgba_prevprev = raw_prevprev[:, :, :4]

            # render from t - 2
            rgb_map_prevprev_dy, _, weights_prevprev_dy = raw2outputs_warp(raw_rgba_prevprev, 
                                                                           z_vals, rays_d, 
                                                                           raw_noise_std)

            ret['rgb_map_pp_dy'] = rgb_map_prevprev_dy

    else:
        # render points at t + 2
        img_idx_rep_postpost = torch.ones_like(pts[:, :, 0:1]) * (img_idx + 2./num_img * 2. )    
        pts_postpost = torch.cat([(pts_post[:, :, :3] + raw_sf_post2postpost), img_idx_rep_postpost] , -1)
        ret['raw_pts_pp'] = pts_postpost[:, :, :3]

        if chain_5frames:
            raw_postpost = network_query_fn(pts_postpost, viewdirs, network_fn)
            raw_rgba_postpost = raw_postpost[:, :, :4]

            # render from t + 2
            rgb_map_postpost_dy, _, weights_postpost_dy = raw2outputs_warp(raw_rgba_postpost, 
                                                                           z_vals, rays_d, 
                                                                           raw_noise_std)

            ret['rgb_map_pp_dy'] = rgb_map_postpost_dy


    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret