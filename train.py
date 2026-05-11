#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from datetime import datetime
import torch
import random
import numpy as np
from random import randint
import yaml
from utils.loss_utils import l1_loss, ssim, get_img_grad_weight, DiffTool, lncc2
from utils.graphics_utils import patch_offsets, patch_warp
from gaussian_renderer import render, network_gui
import sys, time
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import cv2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, erode
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel
from scene.cameras import Camera

import torch.nn.functional as F
# xp debug
import open3d as o3d
from utils.point_utils import depths_to_points_color

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F

def setup_seed(seed):
     random.seed(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     os.environ['PYTHONHASHSEED'] = str(seed)
     
setup_seed(2025)



@torch.no_grad()
def get_confience(opt, iteration, viewpoint_cam, gaussians, depth, render_pkg, nearest_cam, nearest_render_pkg, gt_image_gray, debug_path, if_pho_cfd=False, flag_debug=True):
    patch_size = opt.multi_view_patch_size
    sample_num = opt.multi_view_sample_num
    pixel_noise_th = opt.multi_view_pixel_noise_th
    total_patch_size = (patch_size * 2 + 1) ** 2

    ## compute geometry consistency mask and loss
    H, W = depth.squeeze().shape
    ix, iy = torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([ix, iy], dim=-1).float().to(depth.device)

    pts = gaussians.get_points_from_depth(viewpoint_cam, depth)
    pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
    map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)


    pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
    pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]  
    R = torch.tensor(nearest_cam.R).float().cuda()
    T = torch.tensor(nearest_cam.T).float().cuda()
    pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)  # 转到世界坐标系
    pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]  
    pts_projections = torch.stack(
                [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()  
    pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)  
    if not opt.wo_use_geo_occ_aware:
        d_mask = d_mask & (pixel_noise < pixel_noise_th)  
        weights = (1.0 / torch.exp(pixel_noise)).detach()  
        weights[~d_mask] = 0 
    else:
        d_mask = d_mask
        weights = torch.ones_like(pixel_noise)
        weights[~d_mask] = 0  
    
    cfd_geo = weights.clone().detach()

    if d_mask.sum() > 0 and if_pho_cfd:
        with torch.no_grad():
            ## sample mask
            d_mask = d_mask.reshape(-1)
            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]

            weights = weights.reshape(-1)[valid_indices]  
            ## sample ref frame patch
            pixels = pixels.reshape(-1,2)[valid_indices]  
            offsets = patch_offsets(patch_size, pixels.device)  
            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()  
            
            H, W = gt_image_gray.squeeze().shape
            pixels_patch = ori_pixels_patch.clone()
            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)  
            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size) # [num_valid_points, 49]

            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]  
            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

        ## compute Homography
        ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
        ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

        ref_local_d = render_pkg['rendered_distance'].squeeze()

        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
        H_ref_to_neareast = ref_to_neareast_r[None] - \
            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
        
        ## compute neareast frame patch
        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)  
        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
        _, nearest_image_gray = nearest_cam.get_image()
        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
        

        ncc = lncc2(ref_gray_val, sampled_gray_val)
        H, W = render_pkg['plane_depth'].shape[1:]
        confidence_map = torch.zeros(H, W).cuda().reshape(-1)
        confidence_map[valid_indices] = ncc.reshape(-1) * weights
        confidence_map = confidence_map.reshape((H, W))

        cfd_pho = confidence_map.clone().detach().view(-1)
    else:
        cfd_pho = None

    return cfd_geo, cfd_pho if if_pho_cfd else None


@torch.no_grad()
def fps(points, nsamples):
    """
    points.shape = (b, n, c)
    return indices.shape = (b, nsamples)
    """
    b, n, c = points.shape
    device = points.device
    dis = torch.ones((b, n), device=device) * 1e10
    indices = torch.zeros((b, nsamples), device=device, dtype=torch.long)

    for i in range(1, nsamples):
        cur_index = indices[:, i - 1].view(b, 1, 1).expand(-1, -1, c)
        cur_point = points.gather(1, cur_index)

        temp = (points - cur_point).square().sum(axis=2)
        mask = (temp < dis)
        dis[mask] = temp[mask]

        index = dis.argmax(dim=1)
        dis[list(range(b)), index] = 0
        indices[:, i] = index
    return indices

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, output_path):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # backup main code
    cmd = f'cp ./train.py {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./arguments {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scene {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./utils {dataset.model_path}/'
    os.system(cmd)


    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()


    topk = opt.topk
    fpsk = opt.fpsk
    cddt_k = opt.cddt_k
    max_range = opt.max_range
    diff_tool = DiffTool(max_range=max_range)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    normal_loss, geo_loss, ncc_loss = None, None, None
    rdc_loss = None
    dsmooth_loss = None
    loss_depth=None
    first_iter += 1
    iteration = first_iter
    progress_bar = tqdm(range(first_iter, opt.iterations+1), desc="Training progress")
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    flag_get_psudo_gt = False
    while iteration < opt.iterations+1: #iteration in progress_bar: # range(first_iter, opt.iterations + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

            if iteration >= opt.multi_view_weight_from_iter and iteration % opt.peudo_gt_sep_iter < len(viewpoint_stack):
                flag_get_psudo_gt = True
            else:
                flag_get_psudo_gt = False
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image, gt_image_gray = viewpoint_cam.get_image()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            return_plane=iteration>=opt.single_view_weight_from_iter, return_depth_normal=iteration>=opt.single_view_weight_from_iter)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()
        

        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += opt.scale_loss_weight * min_scale_loss.mean()

        # single-view loss
        if iteration >= opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            if not opt.wo_image_weight:
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            else:
                normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
            loss += (normal_loss)

            image_name = viewpoint_cam.image_name

            plane_depth = render_pkg["plane_depth"]
            prior_depth = viewpoint_cam.vggt_depth.unsqueeze(0)
            W, H = prior_depth.shape[1:]

            edge_mask = None
            h_shift = np.random.randint(0, max_range*2+1) - max_range
            v_shift = np.random.randint(0, max_range*2+1) - max_range
            diff_prior, diff_plane, sum_mask_map = diff_tool.get_diff_map(prior_depth, plane_depth, edge_mask, h_shift, v_shift)


            rank = torch.sign(diff_prior) * (-diff_plane)

            rdc_loss = torch.max(rank + 1e-4, 
                                    torch.zeros_like(rank, device=rank.device)
                                        ).mean() * opt.weight_rdc

            loss += rdc_loss
            
            if viewpoint_cam.mono_normal is not None:
                mono_normal = viewpoint_cam.mono_normal.permute(2, 0, 1)
                mono_normal_error = (1 - (depth_normal * mono_normal).sum(dim=0))[None] # .squeeze()
                dsmooth_loss = weight * mono_normal_error.mean() * opt.weight_normal

                loss += dsmooth_loss

            
        if iteration >= opt.multi_view_weight_from_iter and flag_get_psudo_gt: 
            
            

            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]

            nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, app_model=app_model,
                                return_plane=True, return_depth_normal=False)


            depth_render = render_pkg['plane_depth'].squeeze()
            vggt_depth = viewpoint_cam.vggt_depth
            H, W = vggt_depth.shape

            confidence_map_geo, confidence_map_pho = get_confience(opt, iteration, viewpoint_cam, gaussians, depth_render, render_pkg, 
                nearest_cam, nearest_render_pkg, gt_image_gray, debug_path, if_pho_cfd=True, flag_debug=False)
            

            if confidence_map_pho is not None:

                inds = torch.arange(H * W).to(vggt_depth.device)

                depth_conf = viewpoint_cam.depth_conf.detach() 
                depth_conf_thrsh = opt.depth_conf_thrsh 

                depth_conf_mask = depth_conf > depth_conf_thrsh

                H, W = depth_conf.shape
                inds = torch.arange(H * W).to(depth_conf.device)
                inds = inds[depth_conf_mask.view(-1)]
                conf_mask = confidence_map_pho.view(-1)[inds]
                conf_mask_desc, inds_mask_sort = torch.sort(conf_mask, descending=True)
                inds_pho_sort = inds[inds_mask_sort]

                # indices of pixels with top k multiview confidence
                inds_topk = inds_pho_sort[:topk]

                inds_caddt = inds_pho_sort[topk: cddt_k+topk]
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([iy, ix], dim=-1).float().to(depth_conf.device)

                points_2d = pixels.view((-1, 2))[inds_caddt].unsqueeze(0)
                inds_fps = fps(points_2d, fpsk)[0]
                inds_fps = inds_caddt[inds_fps]  # indices of pixels from fps

                anchor_indices = torch.cat([inds_fps, inds_topk], 0) 


                depth_render_flat = depth_render.view(-1)[anchor_indices]  # render depth of anchors
                vggt_depth_flat = vggt_depth.view(-1)[anchor_indices]  # vggt depth of anchors

                A = torch.stack([vggt_depth_flat, torch.ones_like(vggt_depth_flat)], dim=1)
                ATA = A.t().mm(A)
                ATb = A.t().mm(depth_render_flat.unsqueeze(1))
                solution = torch.linalg.solve(ATA, ATb)
                w, q = solution.squeeze() # w: scale, q: shift

                vggt_depth_aligned = vggt_depth * w + q 
                vggt_depth_flat_aligned = vggt_depth_flat * w + q 

                gap = (vggt_depth_flat_aligned - depth_render_flat).abs()
                gap_full = (vggt_depth_aligned - depth_render).abs()

                k_min = max(6 - iteration // 500, 1)

                ind_closest = anchor_indices[torch.topk(gap, k_min, largest=False)[1]]
                psudo_gt_depth = depth_render.view(-1).clone().detach()
                points_invalid_mask = (gap_full > gap.max()).view(-1)

                vggt_depth_aligned_ = vggt_depth_aligned.view(-1)
                psudo_gt_depth[points_invalid_mask] = (psudo_gt_depth[ind_closest] - vggt_depth_aligned_[ind_closest]).mean()  + vggt_depth_aligned_[points_invalid_mask]              
                psudo_gt_depth = psudo_gt_depth.reshape((H, W)).clone()
                psudo_gt_depth = psudo_gt_depth.detach()
                viewpoint_cam.psudo_gt_depth = psudo_gt_depth

            

        if iteration >= opt.multi_view_weight_from_iter and hasattr(viewpoint_cam, 'psudo_gt_depth'):

            depth_render = render_pkg['plane_depth'].squeeze()
            psudo_gt_depth = viewpoint_cam.psudo_gt_depth.detach()

            depth_conf = viewpoint_cam.depth_conf
            dcf_thrsh_2 = opt.depth_conf_thrsh_2 
            mask_dcf_thrsh_2 = depth_conf > dcf_thrsh_2
            loss_depth = (psudo_gt_depth - depth_render).abs()[mask_dcf_thrsh_2].mean() * opt.weight_depth
        
            loss += loss_depth

        
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log

            dsmooth_loss_log = dsmooth_loss.item() if dsmooth_loss is not None else 0.0
            rdc_loss_log = rdc_loss.item() if rdc_loss is not None else 0.0

            loss_depth_log = loss_depth.item() if loss_depth is not None else 0.


            if iteration % 10 == 0:
                loss_dict = { 
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "smooth_loss": f"{dsmooth_loss_log:.{5}f}",
                    "rank_loss": f"{rdc_loss_log:.{5}f}",
                    "depth_loss": f"{loss_depth_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), app_model)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                    
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)
            
            # multi-view observe trim
            if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(gaussians.get_opacity)
                for view in scene.getTrainCameras():
                    render_pkg_tmp = render(view, gaussians, pipe, bg, app_model=app_model, return_plane=False, return_depth_normal=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    gaussians.prune_points(prune_mask)

            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)

            iteration += 1
    
    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":



    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--config", type=str, default="configs/dtu.yaml")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    setup_seed(2025)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset, opt = lp.extract(args), op.extract(args)
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        dataset.__dict__.update(cfg)
        opt.__dict__.update(cfg)

    training(dataset, opt, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.model_path)

    # All done
    print("\nTraining complete.")
