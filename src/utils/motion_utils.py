import numpy as np
import torch
import torch.nn.functional as F
from numba import jit
from torch_scatter import scatter_min, scatter_max, scatter_sum

@jit(nopython=True)
def _get_motion_features(map_from_init, x_frame, x_center, x_bbox, interpolate, max_length=5):

    # Numba requires specifying list types like this
    x_fwrd_motion = [np.empty((max_length, 4), np.float32) for _ in range(0)] 
    x_bwrd_motion = [np.empty((max_length, 4), np.float32) for _ in range(0)] 
    
    x_fwrd_rel_vel = [np.empty((4,), np.float32) for _ in range(0)] 
    x_bwrd_rel_vel = [np.empty((4,), np.float32) for _ in range(0)] 

    x_fwrd_length = [int(_) for _ in range(0)]
    x_bwrd_length = [int(_) for _ in range(0)]

    x_ignore_traj = [np.array(False) for _ in range(0)]

    x_fwrd_missing = [np.empty((max_length,), np.int32) for _ in range(0)] # Bools cause TypeErrors
    x_bwrd_missing = [np.empty((max_length,), np.int32) for _ in range(0)] # Bools cause TypeErrors

    assert max_length >1

    for cluster_ix in range(map_from_init.max() + 1):
        cluster_mask = map_from_init == cluster_ix
        #has_traj = torch.sum(cluster_mask) > 1:
        if np.sum(cluster_mask) > 1:
            frames = x_frame[cluster_mask].reshape(-1)
            #break

            whs = x_bbox[cluster_mask][:, 2:]
            centers = x_center[cluster_mask]
            
            # Sort elements according to time
            #frames_sorted, ix_sort = torch.sort(frames)
            #assert torch.all(frames == frames_sorted), 'Frames are not sorted'

            # Get forward motion features
            #fwrd_motion = np.full((MAX_LENGTH, 4), fill_value = np.nan, dtype = float)        
            fwrd_motion = np.empty((max_length, 4), np.float32)        
            fwrd_motion[...] = np.nan
            fwrd_mask = (frames.max()- frames)< max_length
            fwrd_frames = frames[fwrd_mask]
            #print(fwrd_frames - fwrd_frames.min())
            idx = (fwrd_frames - fwrd_frames.min()).reshape(-1)
            fwrd_motion[idx] = np.concatenate((centers[fwrd_mask], whs[fwrd_mask]), axis = 1)
            fwrd_length = idx.max() - idx.min() + 1

            if len(idx) >=2:
                #assert (torch.sort(idx)[0] == idx).all()
                x_fwrd_rel = (fwrd_motion[idx[-1]] -  fwrd_motion[idx[-2]]) / (idx[-1] - idx[-2])
            
            else:
                # TODO: This should be done separately for fwrd and bwrd
                x_ignore_traj.append(np.array(True))
                continue

            #fwrd_missing = (fwrd_motion!=fwrd_motion).sum(-1) >0
            fwrd_missing = (np.isnan(fwrd_motion).sum(-1) >0).astype(np.int32)
            nans = fwrd_missing[:fwrd_length] >0
            present_idxs = (~nans).nonzero()[0]
            miss_idxs = nans.nonzero()[0]
            
            if interpolate:
                for i in range(fwrd_motion.shape[1]):
                    fwrd_motion[:fwrd_length, i][nans]= np.interp(miss_idxs, present_idxs, fwrd_motion[:fwrd_length, i][~nans])
        
                assert np.isnan(fwrd_motion[:fwrd_length]).sum() == 0

            # Get backward motion features
            #bwrd_motion = np.full((MAX_LENGTH, 4), fill_value = np.nan, dtype = float)        
            bwrd_motion = np.empty((max_length, 4), np.float32)        
            bwrd_motion[...] = np.nan

            bwrd_mask = (frames-frames.min())< max_length
            bwrd_frames = frames[bwrd_mask]
            idx = (bwrd_frames - bwrd_frames.min()).reshape(-1)
            idx = idx.max() - idx # Invert index
            bwrd_motion[idx] = np.concatenate((centers[bwrd_mask], whs[bwrd_mask]), axis = 1)
            bwrd_length = idx.max() - idx.min() + 1
            last_bwrd_pos = bwrd_motion[idx.max()]
                        
            if len(idx) >=2:
                #assert (torch.sort(idx)[0] == idx).all()
                x_bwrd_rel = (bwrd_motion[idx[0]] -  bwrd_motion[idx[1]]) / (idx[0] - idx[1])
            
            else:
                x_ignore_traj.append(np.array(True))
                continue
            
            
            bwrd_missing = (np.isnan(bwrd_motion).sum(-1) >0).astype(np.int32)
            nans = bwrd_missing[:bwrd_length] >0
            present_idxs = (~nans).nonzero()[0]
            miss_idxs = nans.nonzero()[0]
            
            if interpolate:
                for i in range(bwrd_motion.shape[1]):
                    bwrd_motion[:bwrd_length, i][nans]= np.interp(miss_idxs, present_idxs, bwrd_motion[:bwrd_length, i][~nans])

        
                assert np.isnan(bwrd_motion[:bwrd_length]).sum() == 0
            x_ignore_traj.append(np.array(False))

            # Append everything
            x_fwrd_motion.append(fwrd_motion)
            x_bwrd_motion.append(bwrd_motion)
            
            x_fwrd_rel_vel.append(x_fwrd_rel)
            x_bwrd_rel_vel.append(x_bwrd_rel)

            x_fwrd_length.append(fwrd_length)
            x_bwrd_length.append(bwrd_length)

            x_fwrd_missing.append(fwrd_missing)
            x_bwrd_missing.append(bwrd_missing)
        else:
            x_ignore_traj.append(np.array(True))

    return x_fwrd_rel_vel, x_bwrd_rel_vel, x_fwrd_motion, x_fwrd_length, x_bwrd_motion, x_bwrd_length, x_ignore_traj, x_fwrd_missing, x_bwrd_missing

def reparametrize_motion(x_motion, x_rel_vel, x_length, x_missing, parametrization):
    assert parametrization in ('abs', 'offsets', 'offsets_abs_scale')
    rel_pos = x_motion[:, 1:] - x_motion[:, :-1]
    h_vel = rel_pos[..., -1]
    avg_h_vel = torch.nan_to_num(h_vel, 0).sum(dim=1) / (x_length - 1)

    if parametrization == 'abs':
        return x_motion, x_length, x_missing, avg_h_vel

    assert (torch.abs(get_last_pos(rel_pos, x_length - 1) - x_rel_vel)<1e-3).all(), torch.where((get_last_pos(rel_pos, x_length - 1) != x_rel_vel))
    assert (torch.isnan(rel_pos[..., -1]) == torch.isnan(x_motion[..., 1:, -1])).all()

    if parametrization == 'offsets_abs_scale':
        rel_pos[..., -1] = x_motion[..., 1:, -1]

    return rel_pos, x_length - 1, x_missing[:, 1:], avg_h_vel

def get_last_pos(x_motion, x_length):
    assert x_motion.dim() == 3, x_motion.shape
    assert x_length.dim() == 1, x_length.shape
    assert x_motion.shape[0] == x_length.shape[0], f"{x_length.shape}, {x_motion.shape}"
    last_pos = x_motion.gather(1, x_length[:,None, None].expand(-1, -1, 4) - 1)
    return last_pos[:, 0]

def masked2original(tensor, num_nodes, nodes_mask, fill_value=0):
    assert nodes_mask.shape[0] == num_nodes
    assert nodes_mask.sum() == tensor.shape[0]
    orig_shape = [num_nodes] + list(tensor.shape[1:])
    full_tensor = torch.full(orig_shape, device = tensor.device, dtype=tensor.dtype, fill_value=fill_value)
    full_tensor[nodes_mask] = tensor
    return full_tensor

def compute_vel_feats(fwrd_vel_, bwrd_vel_, batch, edge_index):
    """Compute several velocity-based 'motion continuity' measures based on two track's estimated
    forward and backward velocities"""

    num_nodes = batch.num_nodes
    fwrd_vel = masked2original(fwrd_vel_, num_nodes, ~batch.x_ignore_traj, np.nan)
    bwrd_vel = masked2original(bwrd_vel_, num_nodes, ~batch.x_ignore_traj, np.nan)
    row, col = edge_index
    assert (row < col).all()

    vel_angle = F.cosine_similarity(fwrd_vel[row], -bwrd_vel[col])
    vel_angle = torch.nan_to_num(vel_angle, 0)

    fwrd_norm = fwrd_vel.norm(2, dim=1)
    bwrd_norm = fwrd_vel.norm(2, dim=1)
    norm_sim = F.l1_loss(fwrd_norm[row], bwrd_norm[col], reduction='none')
    norm_sim = torch.nan_to_num(norm_sim, -1)

    t_diff = batch.x_frame_start[col] - batch.x_frame_end[row]
    assert (t_diff>0).all()
    interp_vel = (batch.x_center_start[col] - batch.x_center_end[row]) / t_diff
    interp_vel_norm = interp_vel.norm(2, dim=1)
    interp_vel[..., 2:] = 0
    # TODO: should parametrize if doing w/h vel prediction!!!!

    interp_angle_fwrd = F.cosine_similarity(interp_vel, fwrd_vel[row]) 
    interp_angle_bwrd = F.cosine_similarity(interp_vel, -bwrd_vel[col]) 
    interp_angle = (interp_angle_bwrd + interp_angle_fwrd) / 2
    interp_angle = torch.nan_to_num(interp_angle, -1)

    interp_norm_fwrd = F.l1_loss(fwrd_norm[row], interp_vel_norm, reduction='none')
    interp_norm_bwrd = F.l1_loss(bwrd_norm[col], interp_vel_norm, reduction='none')
    
    interp_norm = torch.nan_to_num(0.5*(interp_norm_fwrd + interp_norm_bwrd), -1)

    return {'interp_angle': interp_angle, 
            'interp_norm': interp_norm,
            #'interp_l1': interp_l1,
            'norm_sim': norm_sim,
            'vel_angle': vel_angle
            }
def ccwh2tlbr(boxes):
    """
    Change box parametrization from (center_x, center_y, width, height) to (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    boxes: Tensor with shape (N, 4)
    """
    assert boxes.dim() == 2 and boxes.shape[-1] == 4
    tl = boxes[:, :2] - 0.5*boxes[:, 2:]
    br = boxes[:, :2]  + 0.5*boxes[:, 2:]
    
    return torch.cat((tl, br), dim = 1)

def compute_iou(output, target):
    """
    GIoU computation, adapted from https://pytorch.org/vision/stable/_modules/torchvision/ops/boxes.html#generalized_box_iou
    output and target are tensors with shape (4, N), representing bounding boxes parametrized by their
    top-left and bottom-right corner.
    """
    assert output.dim() == 2 and output.shape[0] == 4
    assert target.dim() == 2 and target.shape[0] == 4

    x1, y1, x2, y2 = output
    x1g, y1g, x2g, y2g = target 

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(output.device)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    miouk = iouk - ((area_c - unionk) / area_c)

    return iouk, miouk
def compute_giou_fwrd_bwrd_motion_sim(curr_batch, m_pred, edge_index=None):
    """For each edge in the graph, given two trajectories (nodes), 
    find the middle time-point between them, collect the predicted backward/forward position for 
    those trajectories, and compute their Generalized IoU"""

    has_motion_feats = ~curr_batch.x_ignore_traj
    fwrd_pred_, bwrd_pred_ = m_pred
    fwrd_pred_, bwrd_pred_ = fwrd_pred_['pred_pos'].detach(), bwrd_pred_['pred_pos'].detach()
    fwrd_pred = masked2original(fwrd_pred_, curr_batch.num_nodes, has_motion_feats, np.nan)
    bwrd_pred = masked2original(bwrd_pred_, curr_batch.num_nodes, has_motion_feats, np.nan)

    if edge_index is None:
        edge_index = curr_batch.edge_index
    #row, col = raw_batch.edge_index
    row, col = edge_index
    assert (row < col).all()

    m_pred_horizon=fwrd_pred.shape[1]
    t_diff = (curr_batch.x_frame_start[col] - curr_batch.x_frame_end[row]).reshape(-1)
    replace = t_diff == 1
    t_diff_ = torch.minimum(t_diff.float() / 2, torch.as_tensor(m_pred_horizon))
    t_diff_fwrd = torch.ceil(t_diff_).long() - 1
    t_diff_bwrd = torch.maximum(torch.floor(t_diff_).long(), torch.as_tensor(1)) -1

    assert (t_diff_bwrd >=0).all()
    assert (t_diff_fwrd >=0).all()

    # Forward/bwrd pred
    
    fwrd_start_box = fwrd_pred[row].gather(1, t_diff_fwrd[:,None, None].expand(-1, -1, 4))[:, 0]
    bwrd_end_box = bwrd_pred[col].gather(1, t_diff_bwrd[:,None, None].expand(-1, -1, 4))[:, 0]

    _, gious = compute_iou(ccwh2tlbr(fwrd_start_box).T, ccwh2tlbr(bwrd_end_box).T)

    # When the time distance is 1, we instead avg pred + real location for fwr/bwrd
    _, gious_rep_fwrd = compute_iou(ccwh2tlbr(fwrd_start_box[replace]).T, ccwh2tlbr(curr_batch.x_center_start[col][replace]).T)
    _, gious_rep_bwrd = compute_iou(ccwh2tlbr(bwrd_end_box[replace]).T, ccwh2tlbr(curr_batch.x_center_end[row][replace]).T)
    gious[replace] = (gious_rep_bwrd + gious_rep_fwrd) / 2
    #return torch.nan_to_num(gious, -1).cuda()
    return torch.clamp(torch.nan_to_num(gious, -2), min=-2, max=1).cuda()


def torch_get_motion_feats(map_from_init, x_frame_, x_center, x_bbox, max_length, interpolate):
    x_frame = x_frame_.view(-1)
    box_coords = torch.cat((x_center, x_bbox[:, 2:]), dim=1)


    # Strategy: map frames from their value to positions within the trajectory
    # Then directly index an array with shape (num_ids, max_trajectory_length) using those
    num_ids = map_from_init.max() + 1
    count_per_id = scatter_sum(torch.ones_like(x_frame), index=map_from_init, dim=0, dim_size = num_ids)
    keep_traj = count_per_id>1

    # Forward x_motion
    device = box_coords.device
    x_fwrd_bwrd_motion = []
    for mode in ['forward', 'backward']:
        scatter_fn = scatter_max if mode == 'forward' else scatter_min
        last_frame, _ = scatter_fn(x_frame, index=map_from_init, dim=0, dim_size = num_ids)

        # Frame index WITHIN trajectory with positions from 0 until max_length - 1
        if mode == 'forward':
            norm_frame = x_frame - last_frame[map_from_init] + max_length - 1
        
        else:
            norm_frame = -x_frame + last_frame[map_from_init] + max_length - 1
            # e.g. 30, 31, 32, 33, 38 and max_length=4 -->  3, 2, 1, 0

        mask = (norm_frame >= 0)

        # Assign Values
        x_motion_ = torch.full((num_ids, max_length, 4), fill_value = np.nan, device=device)
        x_motion_[map_from_init[mask], norm_frame[mask]] = box_coords[mask] 
        x_motion_ = x_motion_[keep_traj]

        if interpolate: # tricky to vectorize
            # Determine which values need to be interpolated (dont extrapolate starts of trajectories)
            has_nans = torch.isnan(x_motion_)
            cumsum = has_nans[..., 0].cumsum(dim=1)
            track_idx = torch.arange(max_length, device=device).view(-1, max_length).expand(cumsum.shape[0], -1)

            needs_interp = has_nans[..., 0] & ((cumsum - 1) < track_idx)

            # Main strategy: flatten coords columns to make the interpolation 1d
            # then call np.interp on the result and unflatten
            flatten_ = lambda x: x.permute(0, 2, 1).flatten()
            unflatten_ = lambda x: x.reshape(x_motion_.shape[0], 4, max_length).permute(0, 2, 1)

            flat_x_motion = flatten_(x_motion_)
            interp_idxs = torch.where(flatten_(needs_interp[..., None].expand(-1, -1, 4)))[0]
            given_idxs = torch.where(flatten_(~has_nans))[0]
            given_vals = flat_x_motion[given_idxs]

            interp_vals = np.interp(interp_idxs.cpu().numpy(), given_idxs.cpu().numpy(), given_vals.cpu().numpy())
            flat_x_motion[interp_idxs] = torch.as_tensor(interp_vals, device=device, dtype=flat_x_motion.dtype)
            x_motion_ = unflatten_(flat_x_motion)


        x_fwrd_bwrd_motion.append(x_motion_)

        #feat_names = ['x_fwrd_rel_vel', 'x_bwrd_rel_vel', 'x_fwrd_motion', 'x_fwrd_length', 'x_bwrd_motion', 'x_bwrd_length', 'x_ignore_traj', 'x_fwrd_missing', 'x_bwrd_missing']

    return dict(x_fwrd_motion=x_fwrd_bwrd_motion[0],
                x_bwrd_motion=x_fwrd_bwrd_motion[1],
                x_ignore_traj = ~keep_traj,


                # dummy assignments to not break things, but they should not be used
                x_fwrd_rel_vel=torch.empty_like(x_fwrd_bwrd_motion[1][:, 0]),
                x_bwrd_rel_vel=torch.empty_like(x_fwrd_bwrd_motion[1][:, 0]),
                x_bwrd_length = torch.ones_like(x_fwrd_bwrd_motion[1][:, 0], dtype=torch.int),
                x_fwrd_length = torch.ones_like(x_fwrd_bwrd_motion[1][:, 0], dtype=torch.int)
                )