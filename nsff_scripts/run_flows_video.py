import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

from flow_utils import *
import skimage.morphology
import torchvision

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = 'cuda'
VIZ = True

def run_maskrcnn(model, img_path): #, intWidth=1024, intHeight=576):
    import PIL
    threshold = 0.5

    o_image = PIL.Image.open(img_path)

    width, height = o_image.size

    if width > height:
        intWidth = 960
        intHeight = int(round( float(intWidth) / width * height))        
    else:
        intHeight = 960
        intWidth = int(round( float(intHeight) / height * width))        

    print('Semantic Seg Width %d Height %d'%(intWidth, intHeight))

    image = o_image.resize((intWidth, intHeight), PIL.Image.ANTIALIAS)

    image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()

    tenHumans = torch.FloatTensor(intHeight, intWidth).fill_(1.0).cuda()

    objPredictions = model([image_tensor])[0]

    for intMask in range(objPredictions['masks'].size(0)):
        if objPredictions['scores'][intMask].item() > threshold:
            if objPredictions['labels'][intMask].item() == 1: # person
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 4: # motorcycle
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 2: # bicycle
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 8: # truck
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 28: # umbrella
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 17: # cat
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 18: # dog
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 36: # snowboard
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 41: # skateboard
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

    npyMask = skimage.morphology.erosion(tenHumans.cpu().numpy(),
                                         skimage.morphology.disk(1))
    npyMask = ((npyMask < 1e-3) * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return npyMask


def motion_segmentation(basedir, threshold):
    import colmap_read_model as read_model

    points3dfile = os.path.join(basedir, 'sparse/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)

    img_dir = glob.glob(basedir + '/images_*x*')[0]  
    img0 = os.path.join(glob.glob(img_dir)[0], '%05d.png'%0)
    shape_0 = cv2.imread(img0).shape

    resized_height, resized_width = shape_0[0], shape_0[1]

    imdata, perm, img_keys, hwf = load_colmap_data(basedir)
    scale_x, scale_y = resized_width / float(hwf[1]), resized_height / float(hwf[0])

    K = np.eye(3)
    K[0, 0] = hwf[2] * scale_x
    K[0, 2] = hwf[1] / 2. * scale_x
    K[1, 1] = hwf[2] * scale_y
    K[1, 2] = hwf[0] / 2. * scale_y

    xx = range(0, resized_width)
    yy = range(0, resized_height)  # , self.resized_h)
    xv, yv = np.meshgrid(xx, yy)
    p_ref = np.float32(np.stack((xv, yv), axis=-1))
    p_ref_h = np.reshape(p_ref, (-1, 2))
    p_ref_h = np.concatenate((p_ref_h, np.ones((p_ref_h.shape[0], 1))), axis=-1).T

    num_frames = len(perm) #- 1

    save_mask_dir = os.path.join(basedir, 'motion_segmentation')
    os.makedirs(save_mask_dir, exist_ok=True)

    for i in range(0, num_frames): #len(perm) - 1):
        im_prev = imdata[img_keys[perm[max(0, i - 1)]]]
        im_ref = imdata[img_keys[perm[i]]]
        im_post = imdata[img_keys[perm[min(num_frames -1, i + 1)]]]

        print(im_prev.name, im_ref.name, im_post.name)

        T_prev_G = extract_poses(im_prev)        
        T_ref_G = extract_poses(im_ref)
        T_post_G = extract_poses(im_post)

        T_ref2prev = np.dot(T_prev_G, np.linalg.inv(T_ref_G))
        T_ref2post = np.dot(T_post_G, np.linalg.inv(T_ref_G))
        # load optical flow 
        if i == 0:
          fwd_flow, fwd_mask = read_optical_flow(basedir, 
                                       im_ref.name, 
                                       read_fwd=True)
          bwd_flow = np.zeros_like(fwd_flow)
          bwd_mask = np.zeros_like(fwd_mask)

        elif i == num_frames - 1:
          bwd_flow, bwd_mask = read_optical_flow(basedir, 
                                       im_ref.name, 
                                       read_fwd=False)
          fwd_flow = np.zeros_like(bwd_flow)
          fwd_mask = np.zeros_like(bwd_mask)

        else:
          fwd_flow, fwd_mask = read_optical_flow(basedir, 
                                       im_ref.name, 
                                       read_fwd=True)
          bwd_flow, bwd_mask = read_optical_flow(basedir, 
                                       im_ref.name, 
                                       read_fwd=False)

        p_post = p_ref + fwd_flow
        p_post_h = np.reshape(p_post, (-1, 2))
        p_post_h = np.concatenate((p_post_h, np.ones((p_post_h.shape[0], 1))), axis=-1).T

        fwd_e_dist = compute_epipolar_distance(T_ref2post, K, 
                                               p_ref_h, p_post_h)
        fwd_e_dist = np.reshape(fwd_e_dist, (fwd_flow.shape[0], fwd_flow.shape[1]))

        p_prev = p_ref + bwd_flow
        p_prev_h = np.reshape(p_prev, (-1, 2))
        p_prev_h = np.concatenate((p_prev_h, np.ones((p_prev_h.shape[0], 1))), axis=-1).T

        bwd_e_dist = compute_epipolar_distance(T_ref2prev, K, 
                                               p_ref_h, 
                                               p_prev_h)
        bwd_e_dist = np.reshape(bwd_e_dist, (bwd_flow.shape[0], bwd_flow.shape[1]))

        # e_dist = np.maximum(bwd_e_dist, fwd_e_dist)
        # for non-video sequence
        e_dist = np.maximum(bwd_e_dist * bwd_mask, fwd_e_dist * fwd_mask)

        motion_mask = skimage.morphology.binary_opening(e_dist > threshold, skimage.morphology.disk(1))

        cv2.imwrite(os.path.join(save_mask_dir, im_ref.name.replace('.jpg', '.png')), np.uint8(255 * (0. + motion_mask)))

    # RUN SEMANTIC SEGMENTATION
    img_dir = os.path.join(basedir, 'images')
    img_path_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg'))) \
                    + sorted(glob.glob(os.path.join(img_dir, '*.png')))
    semantic_mask_dir = os.path.join(basedir, 'semantic_mask')
    netMaskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()
    os.makedirs(semantic_mask_dir, exist_ok=True)

    for i in range(0, len(img_path_list)):
        img_path = img_path_list[i]
        img_name = img_path.split('/')[-1]
        semantic_mask = run_maskrcnn(netMaskrcnn, 
                                     img_path)
        cv2.imwrite(os.path.join(semantic_mask_dir, 
                                img_name.replace('.jpg', '.png')), 
                    semantic_mask)

    # combine them
    save_mask_dir = os.path.join(basedir, 'motion_masks')
    os.makedirs(save_mask_dir, exist_ok=True)

    mask_dir = os.path.join(basedir, 'motion_segmentation')
    mask_path_list = sorted(glob.glob(os.path.join(mask_dir, '*.png')))

    semantic_dir = os.path.join(basedir, 'semantic_mask')

    for mask_path in mask_path_list:
        print(mask_path)

        motion_mask = cv2.imread(mask_path)
        motion_mask = cv2.resize(motion_mask, (resized_width, resized_height), 
                                interpolation=cv2.INTER_NEAREST) 
        motion_mask = motion_mask[:, :, 0] > 0.1

        # combine from motion segmentation
        semantic_mask = cv2.imread(os.path.join(semantic_dir, mask_path.split('/')[-1]))
        semantic_mask = cv2.resize(semantic_mask, (resized_width, resized_height), 
                                interpolation=cv2.INTER_NEAREST)
        semantic_mask = semantic_mask[:, :, 0] > 0.1
        motion_mask = semantic_mask | motion_mask

        motion_mask = skimage.morphology.dilation(motion_mask, skimage.morphology.disk(2))
        cv2.imwrite(os.path.join(save_mask_dir, '%s'%mask_path.split('/')[-1]), 
                    np.uint8(np.clip((motion_mask), 0, 1) * 255) )
        # cv2.imwrite(os.path.join(mask_img_dir, '%s'%mask_path.split('/')[-1]), np.uint8(np.clip( (1. - motion_mask[..., np.newaxis]) * image, 0, 1) * 255) )

    # delete old mask dir
    os.system('rm -r %s'%mask_dir)
    os.system('rm -r %s'%semantic_dir)


def load_image(imfile):
    long_dim = 768

    img = np.array(Image.open(imfile)).astype(np.uint8)

    # Portrait Orientation
    if img.shape[0] > img.shape[1]:
        input_h = long_dim
        input_w = int(round( float(input_h) / img.shape[0] * img.shape[1]))
    # Landscape Orientation
    else:
        input_w = long_dim 
        input_h = int(round( float(input_w) / img.shape[1] * img.shape[0]))

    print('flow input w %d h %d'%(input_w, input_h))
    img = cv2.resize(img, (input_w, input_h), cv2.INTER_LINEAR)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def load_image_list(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))

    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]
        

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def resize_flow(flow, img_h, img_w):
    # flow = np.load(flow_path)

    flow_h, flow_w = flow.shape[0], flow.shape[1]
    flow[:, :, 0] *= float(img_w)/float(flow_w)
    flow[:, :, 1] *= float(img_h)/float(flow_h)
    flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)

    return flow

def read_img(img_dir, img1_name, img2_name):
    return cv2.imread(os.path.join(img_dir, img1_name + '.png')), \
        cv2.imread(os.path.join(img_dir, img2_name + '.png'))

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:,:,0] += np.arange(w)
    flow_new[:,:,1] += np.arange(h)[:,np.newaxis]

    res = cv2.remap(img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return res

def refinement_flow(fwd_flow, img1, img2):
    flow_refine = cv2.VariationalRefinement.create()

    refine_flow = flow_refine.calc(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 
                                 cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 
                                 fwd_flow)

    return refine_flow

def compute_fwdbwd_mask(fwd_flow, bwd_flow):
    alpha_1 = 0.5
    alpha_2 = 0.5

    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = fwd_lr_error < alpha_1  * (np.linalg.norm(fwd_flow, axis=-1) \
                + np.linalg.norm(bwd2fwd_flow, axis=-1)) + alpha_2

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = bwd_lr_error < alpha_1  * (np.linalg.norm(bwd_flow, axis=-1) \
                + np.linalg.norm(fwd2bwd_flow, axis=-1)) + alpha_2

    return fwd_mask, bwd_mask

def run_optical_flows(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    basedir = "%s"%args.data_path
    # print(basedir)
    img_dir = glob.glob(basedir + '/images_*')[0]  #basedir + '/images_*288'  

    img_path_train = os.path.join(glob.glob(img_dir)[0], '%05d.png'%0)
    img_train = cv2.imread(img_path_train)

    interval = 1    
    of_dir = os.path.join(basedir, 'flow_i%d'%interval)

    if not os.path.exists(of_dir):
        os.makedirs(of_dir)

    with torch.no_grad():
        images = glob.glob(os.path.join(basedir, 'images/', '*.png')) + \
                 glob.glob(os.path.join(basedir, 'images/', '*.jpg'))

        images = load_image_list(images)
        for i in range(images.shape[0]-1):
            print(i)
            image1 = images[i,None]
            image2 = images[i + 1,None]

            _, flow_up_fwd = model(image1, image2, iters=20, test_mode=True)
            _, flow_up_bwd = model(image2, image1, iters=20, test_mode=True)

            flow_up_fwd = flow_up_fwd[0].cpu().numpy().transpose(1, 2, 0)
            flow_up_bwd = flow_up_bwd[0].cpu().numpy().transpose(1, 2, 0)

            img1 = cv2.resize(np.uint8(np.clip(image1[0].cpu().numpy().transpose(1, 2, 0), 0, 255)), 
                             (img_train.shape[1], img_train.shape[0]), 
                              cv2.INTER_LINEAR)
            img2 = cv2.resize(np.uint8(np.clip(image2[0].cpu().numpy().transpose(1, 2, 0), 0, 255)), 
                             (img_train.shape[1], img_train.shape[0]), 
                             cv2.INTER_LINEAR)

            fwd_flow = resize_flow(flow_up_fwd, 
                                   img_train.shape[0], 
                                   img_train.shape[1])
            # fwd_flow = refinement_flow(fwd_flow, img1, img2)

            bwd_flow = resize_flow(flow_up_bwd, 
                                   img_train.shape[0], 
                                   img_train.shape[1])
            # bwd_flow = refinement_flow(bwd_flow, img1, img2)

            fwd_mask, bwd_mask = compute_fwdbwd_mask(fwd_flow, 
                                                     bwd_flow)

            if VIZ:
                if not os.path.exists('./viz_flow'):
                    os.makedirs('./viz_flow')

                if not os.path.exists('./viz_warp_imgs'):
                    os.makedirs('./viz_warp_imgs')

                plt.figure(figsize=(12, 6))
                plt.subplot(2,3,1)
                plt.imshow(img1)
                plt.subplot(2,3,4)
                plt.imshow(img2)

                plt.subplot(2,3,2)
                plt.imshow(flow_viz.flow_to_image(fwd_flow)/255.)
                plt.subplot(2,3,3)
                plt.imshow(flow_viz.flow_to_image(bwd_flow)/255.)

                plt.subplot(2,3,5)
                plt.imshow(flow_viz.flow_to_image(fwd_flow)/255. * np.float32(fwd_mask[..., np.newaxis]))

                plt.subplot(2,3,6)
                plt.imshow(flow_viz.flow_to_image(bwd_flow)/255. * np.float32(bwd_mask[..., np.newaxis]))

                plt.savefig('./viz_flow/%02d.jpg'%i)
                plt.close()

                warped_im2 = warp_flow(img2, fwd_flow)
                warped_im0 = warp_flow(img1, bwd_flow)
  
                cv2.imwrite('./viz_warp_imgs/im_%05d.jpg'%(i), img1[..., ::-1])
                cv2.imwrite('./viz_warp_imgs/im_%05d_fwd.jpg'%(i), warped_im2[..., ::-1])
                cv2.imwrite('./viz_warp_imgs/im_%05d_bwd.jpg'%(i + 1), warped_im0[..., ::-1])

            np.savez(os.path.join(of_dir, '%05d_fwd.npz'%i), flow=fwd_flow, mask=fwd_mask)
            np.savez(os.path.join(of_dir, '%05d_bwd.npz'%(i + 1)), flow=bwd_flow, mask=bwd_mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', 
                        action='store_true', 
                        help='use small model')
    parser.add_argument('--mixed_precision', 
                        action='store_true',
                        help='use mixed precision')
    parser.add_argument("--data_path", type=str, 
                        help='COLMAP Directory')
    parser.add_argument("--epi_threshold", type=float, 
                        default=1.0,
                        help='epipolar distance threshold for physical motion segmentation')
    # parser.add_argument("--input_flow_w", type=int, 
                        # default=768,
                        # help='input image width for optical flow, \
                        # the height will be computed based on original aspect ratio ')

    # parser.add_argument("--input_semantic_w", type=int, 
                        # default=1024,
                        # help='input image width for semantic segmentation')

    # parser.add_argument("--input_semantic_h", type=int, 
                        # default=576,
                        # help='input image height for semantic segmentation')

    args = parser.parse_args()

    run_optical_flows(args)
    motion_segmentation(args.data_path, args.epi_threshold)