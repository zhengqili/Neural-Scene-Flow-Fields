import numpy as np
import os, imageio
import sys
import cv2

########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original


def _load_data(basedir, start_frame, end_frame, 
               factor=None, width=None, height=None, 
               load_imgs=True, evaluation=False):
    print('factor ', factor)
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses_arr = poses_arr[start_frame:end_frame, ...]

    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        # _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(round(sh[1] / factor))
        # width = int((sh[1] / factor))
        # _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        width = int(round(sh[0] / factor))
        # _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgfiles = imgfiles[start_frame:end_frame]

    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), 
                                                                poses.shape[-1]) )
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    if evaluation:
        return poses, bds, imgs,

    disp_dir = os.path.join(basedir, 'disp')  

    dispfiles = [os.path.join(disp_dir, f) \
                for f in sorted(os.listdir(disp_dir)) if f.endswith('npy')]
    dispfiles = dispfiles[start_frame:end_frame]

    disp = [cv2.resize(read_MiDaS_disp(f, 3.0), 
                    (imgs.shape[1], imgs.shape[0]), 
                    interpolation=cv2.INTER_NEAREST) for f in dispfiles]
    disp = np.stack(disp, -1)  

    mask_dir = os.path.join(basedir, 'motion_masks')
    maskfiles = [os.path.join(mask_dir, f) \
                for f in sorted(os.listdir(mask_dir)) if f.endswith('png')]
    maskfiles = maskfiles[start_frame:end_frame]

    masks = [cv2.resize(imread(f)/255., (imgs.shape[1], imgs.shape[0]), 
                        interpolation=cv2.INTER_NEAREST) for f in maskfiles]
    masks = np.stack(masks, -1)  
    masks = np.float32(masks > 1e-3)
    
    # print(masks.shape)
    # sys.exit()

    motion_coords = []
    for i in range(masks.shape[-1]):
        mask = masks[:, :, i]
        coord_y, coord_x = np.where(mask > 0.1)
        coord = np.stack((coord_y, coord_x), -1)
        motion_coords.append(coord)
    
    print(imgs.shape)
    print(disp.shape)

    assert(imgs.shape[0] == disp.shape[0])
    assert(imgs.shape[0] == masks.shape[0])

    assert(imgs.shape[1] == disp.shape[1])
    assert(imgs.shape[1] == masks.shape[1])

    return poses, bds, imgs, disp, masks, motion_coords


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], 
                    np.array([np.cos(theta), 
                              -np.sin(theta), 
                              -np.sin(theta*zrate), 
                              1.]) * rads) 
        
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################
# def spherify_poses(poses, bds):
    
#     p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
#     rays_d = poses[:,:3,2:3]
#     rays_o = poses[:,:3,3:4]

#     def min_line_dist(rays_o, rays_d):
#         A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
#         b_i = -A_i @ rays_o
#         pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
#         return pt_mindist

#     pt_mindist = min_line_dist(rays_o, rays_d)
    
#     center = pt_mindist
#     up = (poses[:,:3,3] - center).mean(0)

#     vec0 = normalize(up)
#     vec1 = normalize(np.cross([.1,.2,.3], vec0))
#     vec2 = normalize(np.cross(vec0, vec1))
#     pos = center
#     c2w = np.stack([vec1, vec2, vec0, pos], 1)

#     poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

#     rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
#     sc = 1./rad
#     poses_reset[:,:3,3] *= sc
#     bds *= sc
#     rad *= sc
    
#     centroid = np.mean(poses_reset[:,:3,3], 0)
#     zh = centroid[2]
#     radcircle = np.sqrt(rad**2-zh**2)
#     new_poses = []
    
#     for th in np.linspace(0.,2.*np.pi, 120):

#         camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
#         up = np.array([0,0,-1.])

#         vec2 = normalize(camorigin)
#         vec0 = normalize(np.cross(vec2, up))
#         vec1 = normalize(np.cross(vec2, vec0))
#         pos = camorigin
#         p = np.stack([vec0, vec1, vec2, pos], 1)

#         new_poses.append(p)

#     new_poses = np.stack(new_poses, 0)
    
#     new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
#     poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
#     return poses_reset, new_poses, bds


def read_MiDaS_disp(disp_fi, disp_rescale=10., h=None, w=None):
    disp = np.load(disp_fi)
    return disp


def load_nvidia_data(basedir, start_frame, end_frame, 
                     factor=8, target_idx=10, 
                     recenter=True, bd_factor=.75, 
                     spherify=False, path_zflat=False,
                     final_height=288):

    poses, bds, imgs = _load_data(basedir, start_frame, end_frame,
                                  height=final_height,
                                  evaluation=True)

    print('Loaded', basedir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], 
                            -poses[:, 0:1, :], 
                            poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(imgs, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(np.percentile(bds[:, 0], 5) * bd_factor)
    # sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc

    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    c2w = poses[target_idx, :, :]
    # c2w = poses_avg(poses)

    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))
    # up = normalize(poses[target_idx, :3, 1])

    # Find a reasonable "focus disp" for this dataset
    close_disp, inf_disp = bds.min()*.9, bds.max()*5.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_disp + dt/inf_disp))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_disp * .1
    tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_views = 120
    N_rots = 2

    if path_zflat:
        zloc = -close_disp * .1
        c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
        rads[2] = 0.
        N_rots = 1
        N_views/=2

    # Generate poses for spiral path
    # render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
    render_poses = render_wander_path(c2w)
    render_poses = np.array(render_poses).astype(np.float32)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses 


def load_llff_data(basedir, start_frame, end_frame, 
                   factor=8, target_idx=10, 
                   recenter=True, bd_factor=.75, 
                   spherify=False, path_zflat=False, 
                   final_height=288):
    
    poses, bds, imgs, disp, masks, motion_coords = _load_data(basedir, 
                                                              start_frame, end_frame,
                                                              height=final_height,
                                                              evaluation=False)
    
    print('Loaded', basedir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], 
                            -poses[:, 0:1, :], 
                            poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(imgs, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    disp = np.moveaxis(disp, -1, 0).astype(np.float32)
    masks = np.moveaxis(masks, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(np.percentile(bds[:, 0], 5) * bd_factor)
    # sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)

    poses[:,:3,3] *= sc

    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    c2w = poses[target_idx, :, :]
    # c2w = poses_avg(poses)

    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))
    # up = normalize(poses[target_idx, :3, 1])

    # Find a reasonable "focus disp" for this dataset
    close_disp, inf_disp = bds.min()*.9, bds.max()*5.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_disp + dt/inf_disp))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_disp * .1
    tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_views = 120
    N_rots = 2

    if path_zflat:
        zloc = -close_disp * .1
        c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
        rads[2] = 0.
        N_rots = 1
        N_views/=2

    # Generate poses for spiral path
    # render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
    render_poses = render_wander_path(c2w)
    render_poses = np.array(render_poses).astype(np.float32)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    disp = disp.astype(np.float32)
    masks = masks.astype(np.float32)

    return images, disp, masks, poses, bds,\
        render_poses, c2w, motion_coords


import torch

def create_bt_poses(hwf):
    num_frames = 40
    max_disp = 32.0 # 64 , 48

    max_trans = max_disp / hwf[2] #self.targets['K_src'][0, 0, 0]  # Maximum camera translation to satisfy max_disp parameter
    z_shift = -max_trans / 6.#-12.0

    print(z_shift)

    init_pos = np.arcsin(-z_shift / max_trans) * float(num_frames) / (2.0 * np.pi)

    max_trans = max_disp / hwf[2] #self.targets['K_src'][0, 0, 0]  # Maximum camera translation to satisfy max_disp parameter
    output_poses = []

    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) /2.0 #* 3.0 / 4.0
        z_trans = 0.#z_shift + max_trans * np.sin(2.0 * np.pi * float(init_pos + i) / float(num_frames))

        i_pose = np.concatenate([
            np.concatenate(
                [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
        ],axis=0)#[np.newaxis, :, :]

        i_pose = np.linalg.inv(i_pose) #torch.tensor(np.linalg.inv(i_pose)).float()
        output_poses.append(i_pose)

    return output_poses

def render_wander_path(c2w):
    hwf = c2w[:,4:5]
    num_frames = 60
    max_disp = 48.0 # 64 , 48

    max_trans = max_disp / hwf[2][0] #self.targets['K_src'][0, 0, 0]  # Maximum camera translation to satisfy max_disp parameter
    output_poses = []

    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) /3.0 #* 3.0 / 4.0
        z_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) /3.0

        i_pose = np.concatenate([
            np.concatenate(
                [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
        ],axis=0)#[np.newaxis, :, :]

        i_pose = np.linalg.inv(i_pose) #torch.tensor(np.linalg.inv(i_pose)).float()

        ref_pose = np.concatenate([c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

        render_pose = np.dot(ref_pose, i_pose)
        # print('render_pose ', render_pose.shape)
        # sys.exit()
        output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
    
    return output_poses