import colmap_read_model as read_model
import numpy as np
import os
import sys
import json

def get_bbox_corners(points):
  lower = points.min(axis=0)
  upper = points.max(axis=0)
  return np.stack([lower, upper])

def filter_outlier_points(points, inner_percentile):
  """Filters outlier points."""
  outer = 1.0 - inner_percentile
  lower = outer / 2.0
  upper = 1.0 - lower
  centers_min = np.quantile(points, lower, axis=0)
  centers_max = np.quantile(points, upper, axis=0)
  result = points.copy()

  too_near = np.any(result < centers_min[None, :], axis=1)
  too_far = np.any(result > centers_max[None, :], axis=1)

  return result[~(too_near | too_far)]

def load_colmap_data(realdir):
    camerasfile = os.path.join(realdir, 'sparse/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h, w, f]).reshape([3,1])

    imagesfile = os.path.join(realdir, 'sparse/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    img_keys = [k for k in imdata]

    print( 'Images #', len(names))
    perm = np.argsort(names)

    points3dfile = os.path.join(realdir, 'sparse/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)

    # extract point 3D xyz
    point_cloud = []
    for key in pts3d:
        point_cloud.append(pts3d[key].xyz)

    point_cloud = np.stack(point_cloud, 0)
    point_cloud = filter_outlier_points(point_cloud, 0.95)

    bounds_mats = []

    upper_bound = 1000
    
    if upper_bound < len(img_keys):
        print("Only keeping " + str(upper_bound) + " images!")

    for i in perm[0:min(upper_bound, len(img_keys))]:
        im = imdata[img_keys[i]]
        print(im.name)
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

        pts_3d_idx = im.point3D_ids
        pts_3d_vis_idx = pts_3d_idx[pts_3d_idx >= 0]

        # 
        depth_list = []
        for k in range(len(pts_3d_vis_idx)):
          point_info = pts3d[pts_3d_vis_idx[k]]
          P_g = point_info.xyz
          P_c = np.dot(R, P_g.reshape(3, 1)) + t.reshape(3, 1)
          depth_list.append(P_c[2])

        zs = np.array(depth_list)
        close_depth, inf_depth = np.percentile(zs, 5), np.percentile(zs, 95)
        bounds = np.array([close_depth, inf_depth])
        bounds_mats.append(bounds)

    w2c_mats = np.stack(w2c_mats, 0)
    # bounds_mats = np.stack(bounds_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    # bbox_corners = get_bbox_corners(point_cloud)
    # also add camera 
    bbox_corners = get_bbox_corners(
                    np.concatenate([point_cloud, c2w_mats[:, :3, 3]], axis=0))
    
    scene_center = np.mean(bbox_corners, axis=0)
    scene_scale = 1.0 / np.sqrt(np.sum((bbox_corners[1] - bbox_corners[0]) ** 2))

    print('bbox_corners ', bbox_corners)
    print('scene_center ', scene_center, scene_scale)

    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], 
                                        [1,1,poses.shape[-1]])], 1)

    # must switch to [-y, x, z] from [x, -y, -z], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], 
                            -poses[:, 2:3, :], 
                            poses[:, 3:4, :], 
                            poses[:, 4:5, :]], 1)
    
    save_arr = []

    for i in range((poses.shape[2])):
        save_arr.append(np.concatenate([poses[..., i].ravel(), bounds_mats[i]], 0))

    save_arr = np.array(save_arr)
    print(save_arr.shape)
    np.save(os.path.join(realdir, 'poses_bounds.npy'), save_arr)
    with open(os.path.join(realdir, 'scene.json'), 'w') as f:
      json.dump({
          'scale': scene_scale,
          'center': scene_center.tolist(),
          'bbox': bbox_corners.tolist(),
      }, f, indent=2)

import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, 
                        help='COLMAP Directory')

    args = parser.parse_args()

    basedir = args.data_path #"/phoenix/S7/zl548/nerf_data/%s/dense"%scene_name
    load_colmap_data(basedir)
    print( 'Done with imgs2poses' )
