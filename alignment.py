import numpy as np
import torch
import torch.nn.functional as F


def align_features(
    feat: torch.Tensor,
    pose_src: dict,
    pose_dst: dict,
    bev_range: float,
    bev_size: int,
) -> torch.Tensor:
    """
    Warp feat from source vehicle's BEV frame into destination vehicle's BEV frame.

    BEV convention: vehicle origin at centre, +x = forward, heading in radians CCW from +x.
    CARLA yaw (degrees, CW from North) must be converted before calling.

    For each normalised pixel in dst, the corresponding src pixel is:
      norm_src = R(heading_dst - heading_src) * norm_dst
               + R_src^T * (t_dst - t_src) / bev_range

    Args:
        feat:      (1, C, H, W) feature map in source vehicle's frame
        pose_src:  {'x', 'y', 'heading'} world frame, heading in radians
        pose_dst:  {'x', 'y', 'heading'} world frame, heading in radians
        bev_range: metres in each direction (±bev_range)
        bev_size:  pixel width/height of BEV grid
    Returns:
        (1, C, H, W) feature map resampled into destination frame
    """
    d_heading = pose_dst['heading'] - pose_src['heading']
    cos_d = float(np.cos(d_heading))
    sin_d = float(np.sin(d_heading))

    dt_x = pose_dst['x'] - pose_src['x']
    dt_y = pose_dst['y'] - pose_src['y']
    cos_s = float(np.cos(-pose_src['heading']))
    sin_s = float(np.sin(-pose_src['heading']))
    t_local_x = cos_s * dt_x - sin_s * dt_y
    t_local_y = sin_s * dt_x + cos_s * dt_y
    t_norm_x = t_local_x / bev_range
    t_norm_y = t_local_y / bev_range

    theta = torch.tensor(
        [[cos_d, -sin_d, t_norm_x],
         [sin_d,  cos_d, t_norm_y]],
        dtype=feat.dtype,
        device=feat.device,
    ).unsqueeze(0)  # (1, 2, 3)

    grid = F.affine_grid(theta, feat.shape, align_corners=False)
    return F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
