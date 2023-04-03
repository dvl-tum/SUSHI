import torch
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_min, scatter_mean, scatter_add
import time


def assign_node_ids(init_node_ids, map_from_init, threshold):
    """
    Node id is the most common id within a node greater than a threshold. Otherwise it is -1 (ambiguous)
    """

    # Normalize y_id between 0 and max_id so that we can feed it to F.one_hot
    id_unique, id_clusters = torch.unique(init_node_ids, return_inverse=True)
    id_clusters = id_clusters.squeeze()

    # One hot encode each label and scatter add. We have node to id occurence map
    id_one_hot = F.one_hot(id_clusters)  # One hot ids
    cluster_total_id = scatter_add(id_one_hot, map_from_init, dim=0)

    # Get the id with most occurences and the total number to calculate the percentage
    cluster_max_total, cluster_max_ix = torch.max(cluster_total_id, dim=1)
    cluster_total = torch.sum(cluster_total_id, dim=1)

    # Find the correspondences of max_ixs in the original y_id labels
    y_id = id_unique[cluster_max_ix]

    # Check if the percentage of an id within a cluster is larger than a threshold. Otherwise, assign -1
    y_id[cluster_max_total / cluster_total < threshold] = -1

    return y_id.view(-1, 1)



def find_graph_time_valid_edges(node_frames, node_frames_mask, depth, frames_per_level, connectivity, pruning_method=['geometry', 'reid', 'reid']):

    """
    Return edge connections of a graph from node features. Edges need to be within a valid time distance.

    node_frames: tuple containing two tensors with shape (num_nodes, 1), indicating the frame number of each node.
    node_frames_mask: binary map contains 1 for occupied frames
    node_reids: torch.tensor with shape (num_nodes, d). ReID features of a node
    top_k_nns: Top k neighbors to be connected
    depth: current graph depth

    edge_ixs: torch.Tensor with shape (2, num_edges) corresponding to the valid edges
    """
    if isinstance(pruning_method, str):
        pruning_method = (depth + 1)*[pruning_method]
    
    assert isinstance(pruning_method, (list, tuple))

    # Ensure that nodes are sorted according to their starting frame
    assert (torch.sort(node_frames[0])[0] == node_frames[0]).all(), "Nodes are NOT sorted by starting frame. Graph was not created properly!"

    edge_ixs = find_time_valid_connections_flowchunk(node_frames=node_frames, depth=depth, frames_per_level=frames_per_level)

    return edge_ixs

def prune_edges(edge_ixs, node_frames, node_reids, top_k_nns, depth, node_feets, graph_pruning_score=None,  pruning_method=['geometry', 'reid', 'reid'],
                reid_sim_fn=F.pairwise_distance):
    """
    Given a set of time-valid edge connections, keep only the Top K-NN neighbors for each node,
    based on a computed similarity score.
    """
    pruning_score = compute_pruning_score(x_reid = node_reids, x_frame = node_frames, x_feet = node_feets, edge_ixs = edge_ixs, pruning_method = pruning_method[depth], 
                                          graph_pruning_score=graph_pruning_score, reid_sim_fn=reid_sim_fn)
    edge_ixs, edge_mask = find_nearest_neighbor_connections(edge_ixs=edge_ixs, pwise_dist=pruning_score, top_k_nns=top_k_nns, num_nodes = node_reids.shape[0])

    return edge_ixs, edge_mask



def find_time_valid_connections_flowchunk(node_frames, depth, frames_per_level):
    """
    All time valid connections of a graph. Nodes from consecutive splits are merged together. This time, non conflicting
    nodes from the same split are also merged together. So lower layer bad decisions can be recovered.
    """

    # Get the start and end frame for each node
    start_frames = node_frames[0]
    end_frames = node_frames[1]
    min_start_frame = torch.min(start_frames)  # Used for normalization during clustering

    coverage = frames_per_level[depth].item()
    if depth==0:
        prev_coverage = 1
    else:
        prev_coverage = frames_per_level[depth-1].item()

    # Get changepoints for the larger clusters - current coverage
    large_clusters = torch.floor((start_frames-min_start_frame)/coverage)  # Cluster the nodes according to their start frames
    large_cluster_changepoints = torch.where(large_clusters[1:] != large_clusters[:-1])[0] + 1
    large_cluster_changepoints = torch.cat((torch.as_tensor([0]).to(node_frames[0].device), large_cluster_changepoints, torch.as_tensor([start_frames.shape[0]]).to(node_frames[0].device)))

    # Sanity check that previous clusters make sense
    assert torch.all(end_frames - start_frames < prev_coverage), "Inconsistency. Previous nodes last longer than the processing splits"
    assert torch.all(large_clusters == torch.floor((end_frames-min_start_frame)/coverage)), "Inconsistency. End frames do not belong to the same clusters"

    # Loop over the large cluster changepoints
    edge_ixs = []
    for large_start_ix, large_end_ix in zip(large_cluster_changepoints[:-1], large_cluster_changepoints[1:]):
        cluster_end_frames = torch.unique(end_frames[large_start_ix:large_end_ix])
        # Loop over the end frames
        for end_f in cluster_end_frames:
            pasts = torch.where(end_frames[large_start_ix:large_end_ix] == end_f)[0] + large_start_ix  # Find the indices of nodes that ends at end_f
            currs = torch.where(start_frames[large_start_ix:large_end_ix] > end_f)[0] + large_start_ix  # Find the indices of nodes that start after end_f
            if pasts.numel() and currs.numel():
                edge_ixs.append(torch.cartesian_prod(pasts, currs))

    if edge_ixs:
        edge_ixs = torch.cat(edge_ixs).T
        # Sanity check - Last frame of the first node should be smaller than the first frame of the second node
        assert torch.all(end_frames[edge_ixs[0]] < start_frames[edge_ixs[1]]), "Edge timepoints violate constraints!"
    else:
        edge_ixs = torch.zeros((2, 0), dtype=torch.int64, device=node_frames[0].device)

    return edge_ixs


def find_conflicting_edge_ixs(node_frames, depth, frames_per_level):
    """
    Create edge connections between nodes that can not be merged together. These edges will be fed into the correlation
    clustering solver with a really high negative cost so that they can never be merged.
    """
    # Get the start and end frame for each node
    start_frames = node_frames[0]
    end_frames = node_frames[1]
    min_start_frame = torch.min(start_frames)  # Used for normalization during clustering

    if depth == 0:
        prev_coverage = 1
    else:
        prev_coverage = frames_per_level[depth - 1].item()

    # Get changepoints for the smaller clusters - prev coverage
    small_clusters = torch.floor((start_frames-min_start_frame)/prev_coverage)
    small_cluster_changepoints = torch.where(small_clusters[1:] != small_clusters[:-1])[0] + 1
    small_cluster_changepoints = torch.cat((torch.as_tensor([0]).to(node_frames[0].device), small_cluster_changepoints, torch.as_tensor([start_frames.shape[0]]).to(node_frames[0].device)))

    # Loop over the large cluster changepoints
    edge_ixs = []
    for start_ix, end_ix in zip(small_cluster_changepoints[:-1], small_cluster_changepoints[1:]):
        nodes = torch.arange(start_ix, end_ix)
        cart_nodes = torch.cartesian_prod(nodes, nodes).T
        cart_nodes = cart_nodes[:, cart_nodes[0] < cart_nodes[1]]

        # Start and end frames
        s = start_frames[cart_nodes[0]].view(-1)
        s2 = start_frames[cart_nodes[1]].view(-1)
        e = end_frames[cart_nodes[0]].view(-1)
        e2 = end_frames[cart_nodes[1]].view(-1)

        # Conditions
        cond1 = torch.logical_and(s2 <= s, e2 >= s)
        cond2 = torch.logical_and(s2 <= e, e2 >= e)
        cond3 = torch.logical_and(s2 >= s, e2 <= e)
        cond = torch.logical_or(torch.logical_or(cond1, cond2), cond3)

        if torch.sum(cond):
            edge_ixs.append(cart_nodes.T[cond])

    if edge_ixs:
        edge_ixs = torch.cat(edge_ixs).T
    else:
        edge_ixs = torch.zeros((2, 0), dtype=torch.int64, device=node_frames[0].device)

    return edge_ixs


def find_nearest_neighbor_connections(edge_ixs, pwise_dist, top_k_nns, reciprocal_k_nns=True, num_nodes=None):
    """Instead of creating a (num_nodes, num_nodes) matrix, rely on scatter_min to compute Top KNN edges. 
    This function is slower for approx num_nodes < 500, but significantly faster and more memory efficient for num_nodes >500
    """
    if edge_ixs.numel():
        if num_nodes is None:
            num_nodes = edge_ixs.max() + 1

        device = pwise_dist.device

        row, col = edge_ixs
        row_, col_ = torch.cat((row, col)), torch.cat((col, row))
        pwise_dist_ =  torch.cat((pwise_dist, pwise_dist))
        row_idxs = torch.arange(num_nodes, device=device, dtype=torch.long)


        # Get the minimum K times
        edge_idxs = []
        orig_edge_idx = [] # Keep track of the indices of edges in the original edge_ixs tensor
                        # to later build the edge_mask tensor 
        for _ in range(top_k_nns):
            vals, idxs = scatter_min(pwise_dist_, row_, dim=0, dim_size=num_nodes)

            # Append only edges that have not yet been selected 
            mask = (vals < np.inf) & (idxs < row_.shape[0])
            
            idxs = idxs[mask]
            pwise_dist_[idxs] = np.inf # set as 'invalid' the edges that were selected

            #edge_idxs.append(torch.stack((row_idxs[mask], col_[idxs])))
            try:
                edge_idxs.append(torch.stack((row_idxs.cpu()[mask.cpu()], col_.cpu()[idxs.cpu()])).cuda())
                orig_edge_idx.append(idxs % row.shape[0]) # Append the edge index, %row.shape[0], because 
                                                    # we are using cat[pwise_dist, pwise_dist_]

            except:
                pass
        edge_idxs = torch.cat(edge_idxs, dim = 1)
        orig_edge_idx = torch.cat(orig_edge_idx)
        edge_idxs, _ = torch.sort(edge_idxs, dim=0) # sort each edge as (i, j) with i<j


        # Get the unique indices from the original set of edges
        orig_edge_idx, edge_counts = torch.unique(orig_edge_idx, return_counts = True)

        # An edge is a reciprocal knn if and only if it apears twice in the raw knn edges
        if reciprocal_k_nns:
            mask = edge_counts == 2
            orig_edge_idx = orig_edge_idx[mask]

        # Select the topknn edges
        final_edges = edge_ixs[:, orig_edge_idx]

        # Build a mask from the indices
        edge_mask = torch.zeros(edge_ixs.shape[1], dtype=torch.bool, device=device)
        edge_mask[orig_edge_idx] = 1
        return final_edges, edge_mask
    
    
def compute_pruning_score(x_reid, x_frame, x_feet, edge_ixs, pruning_method = 'geometry', graph_pruning_score=None,
                          reid_sim_fn=F.pairwise_distance):
    """
    Compute single similarity score to be used for edge pruning from possibly several sources of features
    (feet distance, reid_features, etc.)
    """

    SIGMA_T = 40
    SIGMA_P = 100
    SIGMA_APP = 20

    assert pruning_method in ('geometry', 'reid', 'lpc') or pruning_method.startswith('motion_')

    if pruning_method.startswith('motion_'): # Motio
        # we expect something like # 'motion_01', where the second number indicates the weight
        # used to combine motion and reid
        weight = pruning_method.split('_')[1]
        assert weight.isdigit()
        import math
        weight = float(weight) * math.pow(10, -weight.count('0')) # e.g. '001' --> '0.01'
        
        # TODO: Outsource ReID computation to a function
        app_dist = []
        for i in range(0, edge_ixs.shape[1], 50000):
            app_dist.append(reid_sim_fn(x_reid[edge_ixs[0][i:i + 50000]],
                                                x_reid[edge_ixs[1][i:i + 50000]]).view(-1, 1))

        if app_dist:
            app_dist= torch.cat(app_dist, dim=0).squeeze()
                    
        return 2 - (weight*graph_pruning_score +  (1 - weight)*torch.exp(-app_dist/SIGMA_APP))

    if pruning_method in ('geometry', 'lpc'):        
        start_node_feets, end_node_feets = x_feet
        feet_dist = []
        for i in range(0, edge_ixs.shape[1], 50000):
            feet_dist.append(F.pairwise_distance(end_node_feets[edge_ixs[0][i:i + 50000]],
                                                    start_node_feets[edge_ixs[1][i:i + 50000]]).view(-1, 1))

        feet_dist = torch.cat(feet_dist, dim=0).squeeze()        

        if pruning_method == 'geometry':
            return feet_dist

    if pruning_method == 'reid':
        app_dist = []
        for i in range(0, edge_ixs.shape[1], 50000):
            app_dist.append(reid_sim_fn(x_reid[edge_ixs[0][i:i + 50000]],
                                                x_reid[edge_ixs[1][i:i + 50000]]).view(-1, 1))

        return torch.cat(app_dist, dim=0).squeeze()

    
    if pruning_method == 'lpc':
        app_sim = []
        for i in range(0, edge_ixs.shape[1], 50000):
            app_sim.append(F.cosine_similarity(x_reid[edge_ixs[0][i:i + 50000]],
                                                x_reid[edge_ixs[1][i:i + 50000]]).view(-1, 1))

        app_sim = torch.cat(app_sim, dim=0).squeeze()

        start_frame, end_frame = x_frame
        time_dist = []
        for i in range(0, edge_ixs.shape[1], 50000):
            time_dist.append(torch.abs(start_frame[edge_ixs[1][i:i + 50000]] - end_frame[edge_ixs[0][i:i + 50000]]).view(-1, 1))

        time_dist = torch.cat(time_dist, dim=0).squeeze()
        
        time_sim = torch.exp(-time_dist / SIGMA_T)
        pos_sim = torch.exp(-feet_dist / SIGMA_P)

        sim= (time_sim + pos_sim + app_sim) / 3

        return 1 - sim # top_k_nns expectes a distance score, not a similarity score!

    raise RuntimeError("What kind of pruning are you doing??")



def compute_edge_features(edge_ixs, node_frames, node_bboxes, node_feets, node_reids, fps, motion_feats={},
                          edge_feats_to_use=(
                                             'secs_time_dists', 'norm_feet_x_dists', 'norm_feet_y_dists',
                                             'bb_height_dists', 'bb_width_dists', 'emb_dists'),
                                             reid_sim_fn=F.pairwise_distance):
    """
    Computes a dictionary of edge features among pairs of detections

    Returns:
        Dict where edge key is a string referring to the attr name, and each val is a tensor of shape (num_edges)
        with vals of that attribute for each edge.
    """
    if edge_ixs.numel():

        row, col = edge_ixs

        # Unroll the start and end values
        start_frames = node_frames[0]
        end_frames = node_frames[1]
        start_node_bboxes = node_bboxes[0]
        end_node_bboxes = node_bboxes[1]
        start_node_feets = node_feets[0]
        end_node_feets = node_feets[1]

        # Get values per node
        start_det_times = (start_frames / fps.float()).squeeze()
        end_det_times = (end_frames / fps.float()).squeeze()

        start_bb_width = start_node_bboxes[:, 2]
        start_bb_height = start_node_bboxes[:, 3]
        start_feet_x = start_node_feets[:, 0]
        start_feet_y = start_node_feets[:, 1]

        end_bb_width = end_node_bboxes[:, 2]
        end_bb_height = end_node_bboxes[:, 3]
        end_feet_x = end_node_feets[:, 0]
        end_feet_y = end_node_feets[:, 1]

        epsilon = 1e-20  # Avoid by zero
        feet_vel_x = (end_feet_x - start_feet_x)
        feet_vel_y = (end_feet_y - start_feet_y)

        # normalize by bbox height - instead of pure pixel velocity, normalize it with the size of the det
        cluster_mean_bb_height = (start_bb_height + end_bb_height) / 2
        feet_vel_x /= cluster_mean_bb_height
        feet_vel_y /= cluster_mean_bb_height

        # TODO: manipulate the magnitude if required
        cluster_time_duration = end_det_times - start_det_times
        feet_vel_x /= (cluster_time_duration + epsilon)
        feet_vel_y /= (cluster_time_duration + epsilon)

        unnorm_vel_x = (end_feet_x - start_feet_x) / (cluster_time_duration + epsilon)
        unnorm_vel_y = (end_feet_y - start_feet_y) / (cluster_time_duration + epsilon)

        # Calculate ReID distances. Batch it so that we don't exceed the available memory
        emb_dists = []
        for i in range(0, edge_ixs.shape[1], 50000):
            #emb_dists.append(F.pairwise_distance(node_reids[edge_ixs[0][i:i + 50000]],
            emb_dists.append(reid_sim_fn(node_reids[edge_ixs[0][i:i + 50000]],
                                                 node_reids[edge_ixs[1][i:i + 50000]]).view(-1, 1))
        emb_dists = torch.cat(emb_dists, dim=0).squeeze(1)

        # Create a dictionary from candidates. Whenever there is a distance use the start frame of col and end frame of row.
        # This corresponds to the closest detections between two nodes wrt time.
        mean_bb_heights = (start_bb_height[col] + end_bb_height[row]) / 2  # Normalizing factor
        edge_feats_dict = {'relative_x_vels': feet_vel_x[col] - feet_vel_x[row],
                           'relative_y_vels': feet_vel_y[col] - feet_vel_y[row],
                           'row_x_vels': feet_vel_x[row],
                           'row_y_vels': feet_vel_y[row],
                           'col_x_vels': feet_vel_x[col],
                           'col_y_vels': feet_vel_y[col],
                           'secs_time_dists': start_det_times[col] - end_det_times[row],
                           'norm_feet_x_dists': (start_feet_x[col] - end_feet_x[row]) / mean_bb_heights,
                           'norm_feet_y_dists': (start_feet_y[col] - end_feet_y[row]) / mean_bb_heights,
                           'bb_height_dists': torch.log(start_bb_height[col] / end_bb_height[row]),
                           'bb_width_dists': torch.log(start_bb_width[col] / end_bb_width[row]),
                           'emb_dists': emb_dists}
        edge_feats_dict['vel_dists'] = (edge_feats_dict['relative_x_vels'] ** 2 + edge_feats_dict['relative_y_vels'] ** 2) ** (0.5)
        edge_feats_dict['vel_dists_cosine'] = F.cosine_similarity(torch.cat((edge_feats_dict['row_x_vels'].view(-1, 1), edge_feats_dict['row_y_vels'].view(-1, 1)), dim=1), torch.cat((edge_feats_dict['col_x_vels'].view(-1, 1), edge_feats_dict['col_y_vels'].view(-1, 1)), dim=1))
        edge_feats_dict['unnorm_rel_x'] = unnorm_vel_x[col] - unnorm_vel_x[row]
        edge_feats_dict['unnorm_rel_y'] = unnorm_vel_y[col] - unnorm_vel_y[row]
        edge_feats_dict['unnorm_feet_x_dists'] = (start_feet_x[col] - end_feet_x[row])
        edge_feats_dict['unnorm_feet_y_dists'] = (start_feet_y[col] - end_feet_y[row])
        edge_feats_dict['vel_col_x'] = unnorm_vel_x[col]
        edge_feats_dict['vel_col_y'] = unnorm_vel_y[col]
        edge_feats_dict['vel_row_x'] = unnorm_vel_x[row]
        edge_feats_dict['vel_row_y'] = unnorm_vel_y[row]
        edge_feats_dict['frame_dists'] = edge_feats_dict['secs_time_dists'] * fps.float()

        baseline_vel_x = edge_feats_dict['norm_feet_x_dists'] / edge_feats_dict['secs_time_dists']
        baseline_vel_y = edge_feats_dict['norm_feet_y_dists'] / edge_feats_dict['secs_time_dists']

        vel_row_x = feet_vel_x[row]
        vel_row_y = feet_vel_y[row]
        vel_col_x = feet_vel_x[col]
        vel_col_y = feet_vel_y[col]

        baseline_vel = torch.cat((baseline_vel_x.view(-1, 1), baseline_vel_y.view(-1, 1)), dim=1)
        row_vel = torch.cat((vel_row_x.view(-1, 1), vel_row_y.view(-1, 1)), dim=1)
        col_vel = torch.cat((vel_col_x.view(-1, 1), vel_col_y.view(-1, 1)), dim=1)
        edge_feats_dict['baseline_row_vel_cosine'] = F.cosine_similarity(baseline_vel, row_vel)
        edge_feats_dict['baseline_col_vel_cosine'] = F.cosine_similarity(baseline_vel, col_vel)
        edge_feats_dict['baseline_row_vel_l2'] = F.pairwise_distance(baseline_vel, row_vel)
        edge_feats_dict['baseline_col_vel_l2'] = F.pairwise_distance(baseline_vel, col_vel)
        edge_feats_dict['baseline_clusters_cosine'] = (edge_feats_dict['baseline_row_vel_cosine'] + edge_feats_dict['baseline_col_vel_cosine']) / 2
        edge_feats_dict['baseline_clusters_l2'] = (edge_feats_dict['baseline_row_vel_l2'] + edge_feats_dict[
            'baseline_col_vel_l2']) / 2
        edge_feats_dict['row_col_vel_l2'] = F.pairwise_distance(row_vel, col_vel)
        edge_feats_dict.update(motion_feats)

        # Select and stack certain types
        edge_feats = [edge_feats_dict[feat_names] for feat_names in edge_feats_to_use if feat_names in edge_feats_dict]
        edge_feats = torch.stack(edge_feats).T

    else:
        edge_feats = torch.zeros((0, len(edge_feats_to_use)), dtype=node_reids.dtype, device=edge_ixs.device)

    return edge_feats


def assign_edge_labels(node_ids, edge_ixs, node_frames):
    """
    Same as MPNTrack ground truth. Assigns edge labels (tensor with shape (num_edges,)), with labels defined according
    to the network flow MOT formulation. Only the same ids in the consecutive detections are labeled as positive.
    """
    num_nodes = node_ids.shape[0]
    ids = node_ids.squeeze()
    per_edge_ids = torch.stack([ids[edge_ixs[0]], ids[edge_ixs[1]]])
    same_id = (per_edge_ids[0] == per_edge_ids[1]) & (per_edge_ids[0] != -1)
    same_ids_ixs = torch.where(same_id)
    same_id_edges = edge_ixs.T[same_id].T

    start_frames = node_frames[0]
    end_frames = node_frames[1]

    # For every node, we get the index of the node in the future (resp. past) with the same id that is closest in time
    future_mask = same_id_edges[0] < same_id_edges[1]
    future_dists = (start_frames[same_id_edges[1][future_mask]]-end_frames[same_id_edges[0][future_mask]]).squeeze(dim=1)
    active_fut_edges = scatter_min(future_dists, same_id_edges[0][future_mask], dim=0, dim_size=num_nodes)[1]
    original_node_ixs = torch.cat((same_id_edges[1][future_mask], torch.as_tensor([-1]).to(node_ids.device)))  # -1 at the end for nodes that were not present
    active_fut_edges = original_node_ixs[active_fut_edges]  # Recover the node id of the corresponding
    fut_edge_is_active = active_fut_edges[same_id_edges[0]] == same_id_edges[1]

    # Analogous for past edges
    past_mask = same_id_edges[0] > same_id_edges[1]
    past_dists = (start_frames[same_id_edges[0][past_mask]]-end_frames[same_id_edges[1][past_mask]]).squeeze(dim=1)
    active_past_edges = scatter_min(past_dists, same_id_edges[0][past_mask], dim=0, dim_size=num_nodes)[1]
    original_node_ixs = torch.cat((same_id_edges[1][past_mask], torch.as_tensor([-1]).to(node_ids.device)))  # -1 at the end for nodes that were not present
    active_past_edges = original_node_ixs[active_past_edges]
    past_edge_is_active = active_past_edges[same_id_edges[0]] == same_id_edges[1]

    # Recover the ixs of active edges in the original edge_index tensor o
    active_edge_ixs = same_ids_ixs[0][past_edge_is_active | fut_edge_is_active]
    edge_labels = torch.zeros_like(same_id, dtype=torch.float)
    edge_labels[active_edge_ixs] = 1

    edge_mask = torch.logical_and(ids[edge_ixs[0]] != -1, ids[edge_ixs[1]] != -1)

    return edge_labels, edge_mask


def assign_edge_labels_v2(node_ids, edge_ixs, node_frames):
    """
    Assigns edge labels (tensor with shape (num_edges,)), with labels defined according
    to correlation clustering. All instances of the same identity are connected with a positive edge.
    """

    ids = node_ids.squeeze()
    per_edge_ids = torch.stack([ids[edge_ixs[0]], ids[edge_ixs[1]]])
    same_id = (per_edge_ids[0] == per_edge_ids[1]) & (per_edge_ids[0] != -1)
    edge_labels = torch.zeros_like(same_id, dtype=torch.float)
    edge_labels[same_id] = 1

    return edge_labels


def make_symmetric_edges(edge_ixs, edge_features):
    """
    Make edges symmetric by duplicating them.
    """
    edge_ixs = torch.cat((edge_ixs, torch.stack((edge_ixs[1], edge_ixs[0]))), dim=1)
    edge_features = torch.cat((edge_features, edge_features), dim=0)
    return edge_ixs, edge_features


def to_undirected_graph(mot_graph, attrs_to_update):
    """
    Given a MOTGraph object, it updates its Graph object to make its edges directed (instead of having each edge
    (i, j) appear twice (e.g. (i, j) and (j, i)) it only keeps (i, j) with i <j)
    It averages edge attributes in attrs_to_update accordingly.

    Args:
        mot_graph: MOTGraph object
        attrs_to_update: list/tuple of edge attribute names, that will be averaged over each pair of directed edges
    """

    # Make edges undirected
    sorted_edges, _ = torch.sort(mot_graph.edge_index, dim=0)
    undirected_edges, orig_indices = torch.unique(sorted_edges, return_inverse=True, dim=1)
    assert sorted_edges.shape[1] == 2 * undirected_edges.shape[1], "Some edges were not duplicated"
    mot_graph.edge_index = undirected_edges

    # Average values between each pair of directed edges for all attributes in 'attrs_to_update'
    for attr_name in attrs_to_update:
        if hasattr(mot_graph, attr_name):
            undirected_attr = scatter_mean(getattr(mot_graph, attr_name), orig_indices)
            setattr(mot_graph, attr_name, undirected_attr)


def to_lightweight_graph(mot_graph, attrs_to_del):
    """
    Deletes attributes in mot_graph that are not needed for inference, to save memory
    Args:
        mot_graph: MOTGraph object
        attrs_to_del: tuple/list of attributes to delete

    """
    mot_graph.num_nodes = mot_graph.num_nodes
    mot_graph.node_names = torch.arange(mot_graph.num_nodes).to(mot_graph.edge_preds.device)

    # Delete attributes that are unnecessary for inference
    for attr_name in attrs_to_del:
        if hasattr(mot_graph, attr_name):
            delattr(mot_graph, attr_name)


def to_positive_decision_graph(mot_graph, threshold):
    """
    Keep only hard decisions with the edge predictions and remove the redundant edges
    """
    # Prune edges with low prediction score
    edges_mask = mot_graph.edge_preds >= threshold
    mot_graph.edge_index = mot_graph.edge_index.T[edges_mask].T
    mot_graph.edge_preds = mot_graph.edge_preds[edges_mask]


def iou(boxA, boxB):
    """
    Args:
        boxA: numpy array of bounding boxes with size (N, 4)
        boxB: numpy array of bounding boxes with size (M, 4)

    Returns:
        numpy array of size (N,M), where the (i, j) element is the IoU between the ith box in boxA and jth box in boxB.

    Note: bounding box coordinates are given in format (top, left, bottom, right)
    """
    x11, y11, x12, y12 = np.split(boxA, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxB, 4, axis=1)

    # determine the (x, y)-coordinates of the intersection rectangles
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    # compute the area of intersection rectangles
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou


def iou_pairs(boxA, boxB):
    """
    Args:
        boxA: numpy array of bounding boxes with size (N, 4).
        boxB: numpy array of bounding boxes with size (N, 4)
    Returns:
        numpy array of size (N,), where the ith element is the IoU between the ith box in boxA and boxB.
    Note: bounding box coordinates are given in format (top, left, bottom, right)
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[0], boxB[0])
    yA = np.maximum(boxA[1], boxB[1])
    xB = np.minimum(boxA[2], boxB[2])
    yB = np.minimum(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / (boxAArea + boxBArea - interArea).astype(float)

    return iou
