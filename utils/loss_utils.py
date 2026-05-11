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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
import torchvision.transforms as transforms

transform1 = transforms.CenterCrop((576, 768))
transform2 = transforms.CenterCrop((544, 736))

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def TVLoss(network_output, pred_output, edge_margin=1e-2, margin=1e-4):
    """Total variation loss for a 2D image. input is expected to be of shape (channel, h, w)"""
    h_diff = torch.max((network_output[:, 1:, :] - network_output[:, :-1, :]).abs() - margin,
                       torch.zeros_like(network_output[:, 1:, :])) * ((pred_output[:, 1:, :] - pred_output[:, :-1, :]).abs() < edge_margin).float()
    w_diff = torch.max((network_output[:, :, 1:] - network_output[:, :, :-1]).abs() - margin,
                       torch.zeros_like(network_output[:, :, 1:])) * ((pred_output[:, :, 1:] - pred_output[:, :, :-1]).abs() < edge_margin).float()
    return torch.mean(h_diff) + torch.mean(w_diff)

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def patchify(img, patch_size):
    img = img.unsqueeze(0)
    img = F.unfold(img, patch_size, stride=patch_size)
    img = img.transpose(2, 1).contiguous()
    return img.view(-1, patch_size, patch_size)

def patched_depth_ranking_loss(surf_depth, mono_depth, patch_size=-1, margin=1e-4):
    if patch_size > 0:
        surf_depth_patches = patchify(surf_depth, patch_size).view(-1, patch_size * patch_size) # [N, P*P]
        mono_depth_patches = patchify(mono_depth, patch_size).view(-1, patch_size * patch_size)
    else:
        surf_depth_patches = surf_depth.reshape(-1).unsqueeze(0)
        mono_depth_patches = mono_depth.reshape(-1).unsqueeze(0)

    length = (surf_depth_patches.shape[1]) // 2 * 2
    rand_indices = torch.randperm(length)
    surf_depth_patches_rand = surf_depth_patches[:, rand_indices]
    mono_depth_patches_rand = mono_depth_patches[:, rand_indices]

    patch_rank_loss = torch.max(
        torch.sign(mono_depth_patches_rand[:, :length // 2] - mono_depth_patches_rand[:, length // 2:]) * \
            (surf_depth_patches_rand[:, length // 2:] - surf_depth_patches_rand[:, :length // 2]) + margin,
        torch.zeros_like(mono_depth_patches_rand[:, :length // 2], device=mono_depth_patches_rand.device)
    ).mean()

    return patch_rank_loss

def get_depth_ranking_loss(surf_depth, mono_depth, object_mask=None):
    depth_rank_loss = 0.0

    for transform in [transform1, transform2]:
        surf_depth_crop = transform(surf_depth)
        mono_depth_crop = transform(mono_depth.unsqueeze(0))

        object_mask_crop = None
        if object_mask is not None:
            object_mask_crop = transform(object_mask)
            surf_depth_crop[object_mask_crop.float() < 0.5] = -1e-4
            mono_depth_crop[object_mask_crop.float() < 0.5] = -1e-4

        depth_rank_loss += 0.5 * patched_depth_ranking_loss(surf_depth_crop, mono_depth_crop, patch_size=32)

    return depth_rank_loss

def ssim2(img1, img2, window_size=11):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(0)

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img

def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc  # cc本身代表相似度， 1-cc就代表不相似度
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)   # 所以其实这个PGSR在训练的时候也不会优化那些不相似度特别大的地方是吗，这个0.9估计是通过可视化选出来的
    return ncc, mask

def lncc2(ref, nea): # return confidence
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)

    

    ncc = 1 + cc
    ncc = torch.clamp(ncc, 0.0, 2.0) / 2.   # 范围缩放到0-1之间
    return ncc
    # ncc = torch.mean(ncc, dim=1, keepdim=True)
    # mask = (ncc > 1.1)
    # return ncc, mask




class DiffTool:
    def __init__(self, max_range=16):
        conv_list = [None]  # first element is None, because kernel_size >= 3
        with torch.no_grad():
            for r in range(1, max_range+1):
                kernel_size = r * 2 + 1
                conv = torch.nn.Conv2d(1, 1, kernel_size=kernel_size, padding=(r))
                kernel = torch.zeros((1, 1, kernel_size, kernel_size))
                kernel[0, 0, r, r] = 1.
                conv.weight.data = kernel
                conv.bias.data = torch.tensor([0.])
                conv.requires_grad_(False)
                conv = conv.cuda()
                conv_list.append(conv)

        self.max_range = max_range
        self.conv_list = conv_list


    def get_diff_map(self, map_1, map_2, mask_map, hzt_shift, vtc_shift):
        
        r = max(abs(hzt_shift), abs(vtc_shift))

        if r == 0:
            hzt_shift = 1
            vtc_shift = 0
            r = 1
        elif r > self.max_range:
            hzt_shift = self.max_range
            vtc_shift = self.max_range
            r = self.max_range

        conv_tensor = self.conv_list[r].weight.data
        # conv_tensor.zero_()  # 卷积核所有元素清零

        conv_tensor[0, 0, r+vtc_shift, r+hzt_shift] = -1.

        diff_1 = self.conv_list[r](map_1)
        diff_2 = self.conv_list[r](map_2)

        if mask_map is not None:
            conv_tensor[0, 0, r+vtc_shift, r+hzt_shift] = 1.
            sum_mask_map = self.conv_list[r](mask_map)


        conv_tensor[0, 0, r+vtc_shift, r+hzt_shift] = 0. 
        return diff_1, diff_2, sum_mask_map if mask_map is not None else None

    def get_sum_map(self, map_1, map_2, srange=1, mask_map=None):
        if srange > self.max_range:
            srange = self.max_range

        conv_tensor = self.conv_list[srange].weight.data
        conv_tensor.fill_(1.)

        sum_map_1 = self.conv_list[srange](map_1)
        sum_map_2 = self.conv_list[srange](map_2)

        if mask_map is not None:
            sum_map_mask = self.conv_list[srange](mask_map)

        conv_tensor.fill_(0.)
        conv_tensor[0, 0, srange, srange] = 1.


        return sum_map_1, sum_map_2, None if mask_map is None else torch.max(sum_map_mask, torch.ones_like(sum_map_mask, device=sum_map_mask.device))

    def get_edge_region(self, mask_map):
        """
        Args:
            mask_mask: Tensor (1, H, W), value 0 denotes is edge
        """
        mask_map = 1. - mask_map
        conv_tensor = self.conv_list[1].weight.data
        conv_tensor[0, 0, 0, 1] = 1.
        conv_tensor[0, 0, 1, 0] = 1.
        conv_tensor[0, 0, 1, 2] = 1.
        conv_tensor[0, 0, 2, 1] = 1.

        edge_region = self.conv_list[1](mask_map) > 0.5

        conv_tensor.zero_()
        conv_tensor[0, 0, 1, 1] = 1.

        return edge_region




with torch.no_grad():
    kernelsize=3
    conv = torch.nn.Conv2d(1, 1, kernel_size=kernelsize, padding=(kernelsize//2))
    kernel = torch.tensor([[0.,1.,0.],[1.,1.,1.],[0.,1.,0.]]).reshape(1,1,kernelsize,kernelsize)
    conv.weight.data = kernel #torch.ones((1,1,kernelsize,kernelsize))
    conv.bias.data = torch.tensor([0.])
    conv.requires_grad_(False)
    conv = conv.cuda()
    
def nearMean_map(array, mask, kernelsize=3):
    """ array: (H,W) / mask: (H,W) """
    cnt_map = torch.ones_like(array)

    nearMean_map = conv((array * mask)[None,None])
    cnt_map = conv((cnt_map * mask)[None,None])
    nearMean_map = (nearMean_map / (cnt_map+1e-8)).squeeze()
        
    return nearMean_map