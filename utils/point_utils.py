import torch

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    intrins = view.intrinsic[:3, :3]
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def depths_to_points_color(rays_d, R, T, depth, image, scale=1, mask=None):
    st = int(max(int(scale/2)-1,0))
    depth_view = depth.squeeze()[st::scale,st::scale]
    # rays_d = fov_camera.get_rays(scale=scale)
    depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
    pts = (rays_d * depth_view[..., None]).reshape(-1,3)
    R = torch.tensor(R).float().cuda()
    T = torch.tensor(T).float().cuda()
    pts = (pts-T)@R.transpose(-1,-2)

    colors = image.permute(1, 2, 0).reshape(-1, 3)

    if mask is not None:
        mask = (mask > 0.5).flatten()
        points = points[mask]
        colors = colors[mask]
    return pts, colors

"""
def depths_to_points_color(view, depthmap, image, mask=None):
    assert depthmap.shape == (image.shape[1], image.shape[2]), "Depthmap and image must have the same height and width"
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    intrins = view.intrinsic[:3, :3]
    
    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device=depthmap.device).float(),
        torch.arange(H, device=depthmap.device).float(),
        indexing='xy'
    )
    
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    
    # 计算射线方向
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    
    # 获取射线原点
    rays_o = c2w[:3, 3]
    
    # 计算 3D 点
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    
    # 获取每个点的颜色信息
    # 由于图像的形状是 [C, W, H]，需要先转置为 [H, W, C]
    colors = image.permute(1, 2, 0).reshape(-1, image.shape[0])

    if mask is not None:
        mask = (mask > 0.5).flatten()
        points = points[mask]
        colors = colors[mask]

    
    return points, colors
"""
