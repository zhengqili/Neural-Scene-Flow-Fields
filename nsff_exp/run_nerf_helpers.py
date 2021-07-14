import torch
torch.autograd.set_detect_anomaly(False)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
# TODO: remove this dependency
# from torchsearchsorted import searchsorted

# Misc
prob2weights = lambda x: x 

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2mae = lambda x, y : torch.mean(torch.abs(x - y))

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        
        self.sf_linear = nn.Linear(W, 6)
        self.prob_linear = nn.Linear(W, 2)
        # self.blend_linear = nn.Linear(W // 2, 1)

    def forward(self, x):

        if self.use_viewdirs:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x #torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        sf = nn.functional.tanh(self.sf_linear(h))
        prob = nn.functional.sigmoid(self.prob_linear(h))

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            
            # blend_w = nn.functional.sigmoid(self.blend_linear(h))

            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return torch.cat([outputs, sf, prob], dim=-1)



# Model
class Rigid_NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
        """ 
        """
        super(Rigid_NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
    
        self.w_linear = nn.Linear(W, 1)
        # h = F.relu(h)
        # blend_w = nn.functional.sigmoid(self.w_linear(h))

    def forward(self, x):

        if self.use_viewdirs:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x #torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        v = nn.functional.sigmoid(self.w_linear(h))

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return torch.cat([outputs, v], -1)

# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


# def get_rays_np(H, W, focal, c2w):
#     i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
#     dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
#     # Rotate ray directions from camera frame to the world frame
#     rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # Translate camera frame's origin to the world frame. It is the origin of all rays.
#     rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
#     return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]


    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# # Hierarchical sampling (section 5.2)
# def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
#     # Get pdf
#     weights = weights + 1e-5 # prevent nans
#     pdf = weights / torch.sum(weights, -1, keepdim=True)
#     cdf = torch.cumsum(pdf, -1)
#     cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

#     # Take uniform samples
#     if det:
#         u = torch.linspace(0., 1., steps=N_samples)
#         u = u.expand(list(cdf.shape[:-1]) + [N_samples])
#     else:
#         u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

#     # Pytest, overwrite u with numpy's fixed random numbers
#     if pytest:
#         np.random.seed(0)
#         new_shape = list(cdf.shape[:-1]) + [N_samples]
#         if det:
#             u = np.linspace(0., 1., N_samples)
#             u = np.broadcast_to(u, new_shape)
#         else:
#             u = np.random.rand(*new_shape)
#         u = torch.Tensor(u)

#     # Invert CDF
#     u = u.contiguous()
#     inds = searchsorted(cdf, u, side='right')
#     below = torch.max(torch.zeros_like(inds-1), inds-1)
#     above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
#     inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

#     # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
#     # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
#     matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
#     cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
#     bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

#     denom = (cdf_g[...,1]-cdf_g[...,0])
#     denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
#     t = (u-cdf_g[...,0])/denom
#     samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

#     return samples



def compute_depth_loss(pred_depth, gt_depth):   
    # pred_depth_e = NDC2Euclidean(pred_depth_ndc)
    t_pred = torch.median(pred_depth)
    s_pred = torch.mean(torch.abs(pred_depth - t_pred))

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))

    pred_depth_n = (pred_depth - t_pred)/s_pred
    gt_depth_n = (gt_depth - t_gt)/s_gt

    # return torch.mean(torch.abs(pred_depth_n - gt_depth_n))
    return torch.mean(torch.pow(pred_depth_n - gt_depth_n, 2))


def compute_mse(pred, gt, mask, dim=2):
    if dim == 1:
        mask_rep = torch.squeeze(mask)
    if dim == 2:
        mask_rep = mask.repeat(1, pred.size(-1))
    elif dim == 3:
        mask_rep = mask.repeat(1, 1, pred.size(-1))

    num_pix = torch.sum(mask_rep) + 1e-8
    return torch.sum( (pred - gt)**2 * mask_rep )/ num_pix

def compute_mae(pred, gt, mask, dim=2):
    if dim == 1:
        mask_rep = torch.squeeze(mask)
    if dim == 2:
        mask_rep = mask.repeat(1, pred.size(-1))
    elif dim == 3:
        mask_rep = mask.repeat(1, 1, pred.size(-1))

    num_pix = torch.sum(mask_rep) + 1e-8
    return torch.sum( torch.abs(pred - gt) * mask_rep )/ num_pix


# def compute_depth_loss_mask(pred_depth, gt_depth, mask): 

#     mask = torch.squeeze(mask)

#     t_pred = torch.median(pred_depth)
#     s_pred = torch.mean(torch.abs(pred_depth - t_pred))

#     t_gt = torch.median(gt_depth)
#     s_gt = torch.mean(torch.abs(gt_depth - t_gt))

#     pred_depth_n = (pred_depth - t_pred)/s_pred
#     gt_depth_n = (gt_depth - t_gt)/s_gt

#     num_pixe = torch.sum(mask) + 1e-8

#     return torch.sum(torch.abs(pred_depth_n - gt_depth_n) * mask)/num_pixe


def normalize_depth(depth):
    # depth_sm = depth - torch.min(depth)
    return torch.clamp(depth/percentile(depth, 97), 0., 1.)


def percentile(t, q):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def make_color_wheel():
  """
  Generate color wheel according Middlebury color code
  :return: Color wheel
  """
  RY = 15
  YG = 6
  GC = 4
  CB = 11
  BM = 13
  MR = 6

  ncols = RY + YG + GC + CB + BM + MR

  colorwheel = np.zeros([ncols, 3])

  col = 0

  # RY
  colorwheel[0:RY, 0] = 255
  colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
  col += RY

  # YG
  colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
  colorwheel[col:col+YG, 1] = 255
  col += YG

  # GC
  colorwheel[col:col+GC, 1] = 255
  colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
  col += GC

  # CB
  colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
  colorwheel[col:col+CB, 2] = 255
  col += CB

  # BM
  colorwheel[col:col+BM, 2] = 255
  colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
  col += + BM

  # MR
  colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
  colorwheel[col:col+MR, 0] = 255

  return colorwheel


def compute_color(u, v):
  """
  compute optical flow color map
  :param u: optical flow horizontal map
  :param v: optical flow vertical map
  :return: optical flow in color code
  """
  [h, w] = u.shape
  img = np.zeros([h, w, 3])
  nanIdx = np.isnan(u) | np.isnan(v)
  u[nanIdx] = 0
  v[nanIdx] = 0

  colorwheel = make_color_wheel()
  ncols = np.size(colorwheel, 0)

  rad = np.sqrt(u**2+v**2)

  a = np.arctan2(-v, -u) / np.pi

  fk = (a+1) / 2 * (ncols - 1) + 1

  k0 = np.floor(fk).astype(int)

  k1 = k0 + 1
  k1[k1 == ncols+1] = 1
  f = fk - k0

  for i in range(0, np.size(colorwheel,1)):
    tmp = colorwheel[:, i]
    col0 = tmp[k0-1] / 255
    col1 = tmp[k1-1] / 255
    col = (1-f) * col0 + f * col1

    idx = rad <= 1
    col[idx] = 1-rad[idx]*(1-col[idx])
    notidx = np.logical_not(idx)

    col[notidx] *= 0.75
    img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

  return img


def flow_to_image(flow, display=False):
  """
  Convert flow into middlebury color code image
  :param flow: optical flow map
  :return: optical flow image in middlebury color
  """
  UNKNOWN_FLOW_THRESH = 100
  u = flow[:, :, 0]
  v = flow[:, :, 1]

  maxu = -999.
  maxv = -999.
  minu = 999.
  minv = 999.

  idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
  u[idxUnknow] = 0
  v[idxUnknow] = 0

  maxu = max(maxu, np.max(u))
  minu = min(minu, np.min(u))

  maxv = max(maxv, np.max(v))
  minv = min(minv, np.min(v))

  # sqrt_rad = u**2 + v**2
  rad = np.sqrt(u**2 + v**2)

  maxrad = max(-1, np.max(rad))

  if display:
    print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

  u = u/(maxrad + np.finfo(float).eps)
  v = v/(maxrad + np.finfo(float).eps)

  img = compute_color(u, v)

  idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
  img[idx] = 0

  return np.uint8(img)

# NOTE: WE DO IN COLMAP/OPENCV FORMAT, BUT INPUT IS OPENGL FORMAT!!!!!1
def perspective_projection(pts_3d, h, w, f):
    pts_2d = torch.cat([pts_3d[..., 0:1] * f/-pts_3d[..., 2:3] + w/2., 
                        -pts_3d[..., 1:2] * f/-pts_3d[..., 2:3] + h/2.], dim=-1)

    return pts_2d    


def se3_transform_points(pts_ref, raw_rot_ref2prev, raw_trans_ref2prev):
    pts_prev = torch.squeeze(torch.matmul(raw_rot_ref2prev, pts_ref[..., :3].unsqueeze(-1)) + raw_trans_ref2prev)
    return pts_prev


def NDC2Euclidean(xyz_ndc, H, W, f):
    z_e = 2./ (xyz_ndc[..., 2:3] - 1. + 1e-6)
    x_e = - xyz_ndc[..., 0:1] * z_e * W/ (2. * f)
    y_e = - xyz_ndc[..., 1:2] * z_e * H/ (2. * f)

    xyz_e = torch.cat([x_e, y_e, z_e], -1)
 
    return xyz_e

import sys


def projection_from_ndc(c2w, H, W, f, weights_ref, raw_pts, n_dim=1):
    R_w2c = c2w[:3, :3].transpose(0, 1)
    t_w2c = -torch.matmul(R_w2c, c2w[:3, 3:])

    pts_3d = torch.sum(weights_ref[...,None] * raw_pts, -2)  # [N_rays, 3]

    pts_3d_e_world = NDC2Euclidean(pts_3d, H, W, f)

    if n_dim == 1:
        pts_3d_e_local = se3_transform_points(pts_3d_e_world, 
                                              R_w2c.unsqueeze(0), 
                                              t_w2c.unsqueeze(0))
    else:
        pts_3d_e_local = se3_transform_points(pts_3d_e_world, 
                                              R_w2c.unsqueeze(0).unsqueeze(0), 
                                              t_w2c.unsqueeze(0).unsqueeze(0))

    pts_2d = perspective_projection(pts_3d_e_local, H, W, f)

    return pts_2d

def compute_optical_flow(pose_post, pose_ref, pose_prev, H, W, focal, ret, n_dim=1):
    pts_2d_post = projection_from_ndc(pose_post, H, W, focal, 
                                      ret['weights_ref_dy'], 
                                      ret['raw_pts_post'], 
                                      n_dim)
    pts_2d_prev = projection_from_ndc(pose_prev, H, W, focal, 
                                      ret['weights_ref_dy'], 
                                      ret['raw_pts_prev'], 
                                      n_dim)

    return pts_2d_post, pts_2d_prev

def read_optical_flow(basedir, img_i, start_frame, fwd):
    import os
    flow_dir = os.path.join(basedir, 'flow_i1')

    if fwd:
      fwd_flow_path = os.path.join(flow_dir, 
                                  '%05d_fwd.npz'%(start_frame + img_i))
      fwd_data = np.load(fwd_flow_path)#, (w, h))
      fwd_flow, fwd_mask = fwd_data['flow'], fwd_data['mask']
      fwd_mask = np.float32(fwd_mask)  
      
      return fwd_flow, fwd_mask
    else:

      bwd_flow_path = os.path.join(flow_dir, 
                                  '%05d_bwd.npz'%(start_frame + img_i))

      bwd_data = np.load(bwd_flow_path)#, (w, h))
      bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
      bwd_mask = np.float32(bwd_mask)

      return bwd_flow, bwd_mask


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:,:,0] += np.arange(w)
    flow_new[:,:,1] += np.arange(h)[:,np.newaxis]

    res = cv2.remap(img, flow_new, None, 
                    cv2.INTER_CUBIC, 
                    borderMode=cv2.BORDER_CONSTANT)
    return res

def compute_sf_sm_loss(pts_1_ndc, pts_2_ndc, H, W, f):
    # sigma = 2.
    n = pts_1_ndc.shape[1]

    pts_1_ndc_close = pts_1_ndc[..., :int(n * 0.95), :]
    pts_2_ndc_close = pts_2_ndc[..., :int(n * 0.95), :]

    pts_3d_1_world = NDC2Euclidean(pts_1_ndc_close, H, W, f)
    pts_3d_2_world = NDC2Euclidean(pts_2_ndc_close, H, W, f)
        
    # dist = torch.norm(pts_3d_1_world[..., :-1, :] - pts_3d_1_world[..., 1:, :], 
                      # dim=-1, keepdim=True)
    # weights = torch.exp(-dist * sigma).detach()

    # scene flow 
    scene_flow_world = pts_3d_1_world - pts_3d_2_world

    return torch.mean(torch.abs(scene_flow_world[..., :-1, :] - scene_flow_world[..., 1:, :]))

# Least kinetic motion prior
def compute_sf_lke_loss(pts_ref_ndc, pts_post_ndc, pts_prev_ndc, H, W, f):
    n = pts_ref_ndc.shape[1]

    pts_ref_ndc_close = pts_ref_ndc[..., :int(n * 0.9), :]
    pts_post_ndc_close = pts_post_ndc[..., :int(n * 0.9), :]
    pts_prev_ndc_close = pts_prev_ndc[..., :int(n * 0.9), :]

    pts_3d_ref_world = NDC2Euclidean(pts_ref_ndc_close, 
                                     H, W, f)
    pts_3d_post_world = NDC2Euclidean(pts_post_ndc_close, 
                                     H, W, f)
    pts_3d_prev_world = NDC2Euclidean(pts_prev_ndc_close, 
                                     H, W, f)
    
    # scene flow 
    scene_flow_w_ref2post = pts_3d_post_world - pts_3d_ref_world
    scene_flow_w_prev2ref = pts_3d_ref_world - pts_3d_prev_world

    return 0.5 * torch.mean((scene_flow_w_ref2post - scene_flow_w_prev2ref) ** 2)