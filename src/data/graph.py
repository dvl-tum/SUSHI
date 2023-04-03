import torch
from torch_geometric.data import Data
import numpy as np
from src.utils.graph_utils import find_graph_time_valid_edges, prune_edges, compute_edge_features, assign_edge_labels, make_symmetric_edges, assign_node_ids
from src.utils.motion_utils import  torch_get_motion_feats
from torch_scatter import scatter_mean, scatter_max, scatter_min, scatter_add
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import time

class Graph(Data):
    """
    This is the class we use to instantiate our graph objects. We inherit from torch_geometric's Data class and add a
    few convenient methods to it, mostly related to changing data types in a single call. This class correpsonds to
    a graph of a single layer
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _change_attrs_types(self, attr_change_fn):
        """
        Base method for all methods related to changing attribute types. Iterates over the attributes names in
        _data_attr_names, and changes its type via attr_change_fun

        Args:
            attr_change_fn: callable function to change a variable's type
        """
        # These are our standard 'data-related' attribute names.
        _data_attr_names = ['x', # Node feature vecs
                            'x_track', # Node track features
                           'edge_attr', # Edge Feature vecs
                           'edge_index', # Sparse Adjacency matrix
                           'node_names', # Node names (integer values)
                           'edge_labels', # Edge labels according to Network Flow MOT formulation
                           'edge_preds', # Predicted approximation to edge labels
                           'reid_emb_dists', # Reid distance for each edge
                           'conf_edge_index',
                           'edge_mask'] # Conflicting edge indices

        for attr_name in _data_attr_names:
            if hasattr(self, attr_name):
                if getattr(self, attr_name ) is not None:
                    old_attr_val = getattr(self, attr_name)
                    setattr(self, attr_name, attr_change_fn(old_attr_val))

    def tensor(self):
        self._change_attrs_types(attr_change_fn= torch.tensor)
        return self

    def float(self):
        self._change_attrs_types(attr_change_fn= lambda x: x.float())
        return self

    def numpy(self):
        self._change_attrs_types(attr_change_fn= lambda x: x if isinstance(x, np.ndarray) else x.detach().cpu().numpy())
        return self

    def cpu(self):
        #self.tensor()
        self._change_attrs_types(attr_change_fn= lambda x: x.cpu())
        return self

    def cuda(self):
        #self.tensor()
        self._change_attrs_types(attr_change_fn=lambda x: x.cuda())
        return self

    def to(self, device):
        self._change_attrs_types(attr_change_fn=lambda x: x.to(device))

    def device(self):
        if isinstance(self.edge_index, torch.Tensor):
            return self.edge_index.device

        return torch.device('cpu')


class HierarchicalGraph(Data):
    """
    Processes the scene data in a hierarchical manner. Responsible from bookkeeping and creating new graphs for deeper
    layers.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # One-hot encode the frame numbers
        if hasattr(self, 'x_frame'):
            self.x_one_hot_frame = self._one_hot_encode_frames()

        self.curr_depth = torch.tensor(0)  # Current hierarchical

        # Mapping of nodes from prev layers to the deeper
        self.maps = []  # All maps
        self.map_from_init = None  # From initial nodes to the latest layer

    def _one_hot_encode_frames(self):
        """
        Frame numbers are shifted by the start frame and one-hot encoded by this function. These frame numbers are later used to check
        temporal conflicts between clusters in a vectorized manner.
        """
        norm_frames = self.x_frame.view(-1) - self.start_frame
        return F.one_hot(norm_frames, num_classes=self.frames_per_level[-1])

    def _get_curr_graph_specs(self, config):
        """
         Load the specs that will be used for the current layer of the hierarchical graph. In the deeper layers
         we fuse the features of the nodes that merge together.

         x_node: Node features for graph
         x_reid: Reid features for edge distance calculation

         x_frame: Minimum and maximum timepoints that a node represents. Represented by a tuple. First element is the
         x_frame_mask: Binary tensor that represents timepoints occupied by a cluster 
         first frames of the nodes and the second element is the last frame of a node

         x_bbox: Bboxes of the minimum and max initial nodes for each node - Represented by a tuple
         x_feet: Bbox feet of the minimum and max initial nodes for each node - Represented by a tuple

         y_id: Identity of each node. Obtained via a heuristic defined on the ids of each participant of a node
        """

        if self.curr_depth == 0:
            x_node = self.x_node
            x_reid = self.x_reid
            x_frame = (self.x_frame, self.x_frame)
            x_frame_mask = None
            x_bbox = (self.x_bbox, self.x_bbox)
            x_center = (self.x_center, self.x_center)
            x_feet = (self.x_feet, self.x_feet)
            y_id = self.y_id
        else:
            # Node and reid features are the average of those from the initial layer
            x_node = scatter_mean(self.x_node, self.map_from_init, dim=0)
            x_reid = scatter_mean(self.x_reid, self.map_from_init, dim=0)

            # Frame, bbox and feet are obtained by finding the min and max nodes and getting the features from those
            min_frame, ix_min = scatter_min(self.x_frame, self.map_from_init, dim=0)
            max_frame, ix_max = scatter_max(self.x_frame, self.map_from_init, dim=0)
            ix_max, ix_min = ix_max.squeeze(), ix_min.squeeze()

            x_frame = (min_frame, max_frame)
            x_frame_mask = None
            x_bbox = (self.x_bbox[ix_min], self.x_bbox[ix_max])
            x_feet = (self.x_feet[ix_min], self.x_feet[ix_max])
            x_center = (self.x_center[ix_min], self.x_center[ix_max])

            # Obtain new node ids from initials
            y_id = assign_node_ids(init_node_ids=self.y_id, map_from_init=self.map_from_init,
                                   threshold=config.node_id_min_ratio)

        return x_node, x_reid, x_frame, x_frame_mask, x_bbox, x_feet, x_center, y_id

    def _get_track_features(self, config):
        # init x_track and x_track_length
        x_track = []
        x_track_length = []

        # Iterate over every cluster
        for cluster_ix in range(0, torch.max(self.map_from_init).item()+1):

            # Mask the elements of the cluster
            cluster_mask = self.map_from_init == cluster_ix

            if torch.sum(cluster_mask) > 1:
                frames = self.x_frame[cluster_mask]
                whs = self.x_bbox[cluster_mask][:, 2:]
                feets = self.x_feet[cluster_mask]

                # Sort elements according to time
                frames_sorted, ix_sort = torch.sort(frames)
                assert torch.all(frames == frames_sorted), 'Frames are not sorted'

                # Get the velocity and bbox ratios
                velocity = (self.fps * (feets[1:] - feets[:-1])) / (frames[1:] - frames[:-1])
                ratios = whs[1:] / whs[:-1]
                average_bbox_h = (whs[1:, 1] + whs[:-1, 1]) / 2
                norm_velocity = velocity / average_bbox_h.view(-1, 1)
                norm_ratios = torch.log(ratios)


                # Combine all features
                track = torch.cat((norm_velocity, norm_ratios), dim=1)
                # track = velocity

            else:
                track = torch.zeros((1, 4), dtype=self.x_node.dtype, device=self.x_node.device)

            track_length = track.shape[0]

            # Update the lists
            x_track.append(track)
            x_track_length.append(track_length)

        # Pad the tracks
        x_track = pad_sequence(x_track, batch_first=True, padding_value=0)  # Shape (num_nodes, track_length, 4)
        x_track_length = torch.tensor(x_track_length)

        return x_track, x_track_length

    def get_motion_features(self, max_length, interpolate):
        """ Wrapper function around the actual numba code that gathers the trajectory information
        of each node in the current graph
        """
        motion_feats = torch_get_motion_feats(map_from_init=self.map_from_init,
                                              x_frame_=self.x_frame,
                                              x_center=self.x_center,
                                              x_bbox=self.x_bbox,
                                              max_length=max_length,
                                              interpolate=interpolate)
        
        
        have_non_singleton = not motion_feats['x_ignore_traj'].all()
        if not have_non_singleton:
            device = self.x_frame.device
            motion_feats = {'x_fwrd_motion': torch.zeros((0, max_length, 4), device=device),
                            'x_bwrd_motion': torch.zeros((0, max_length, 4), device=device),
                            'x_ignore_traj': motion_feats['x_ignore_traj']
                            }

        return motion_feats
        


    def construct_curr_graph_nodes(self, config):
        # Get current level features and labels
        x_node, x_reid, x_frame, x_frame_mask, x_bbox, x_feet, x_center, y_id = self._get_curr_graph_specs(config)

        #if config.zero_nodes:
        if config.zero_nodes:
            if config.do_hicl_feats and self.curr_depth == 0:
                x_node = torch.zeros_like(x_node)
            
            elif not config.do_hicl_feats:
                x_node = torch.zeros_like(x_node)

        
        if self.curr_depth >0 and config.do_motion:
            motion_features = self.get_motion_features(max_length = config.motion_max_length[self.curr_depth - 1],
                                                       interpolate =config.interpolate_motion)
            fwrd_vel = torch.zeros(((~motion_features['x_ignore_traj']).sum(), 4), device = x_node.device, dtype=torch.float)
            bwrd_vel = fwrd_vel.clone()
                
        else:
            motion_features = {}
            fwrd_vel, bwrd_vel= None, None

        raw_edge_index = find_graph_time_valid_edges(node_frames=x_frame, node_frames_mask=x_frame_mask, 
                                                     depth=self.curr_depth, frames_per_level=self.frames_per_level, 
                                                     connectivity=config.connectivity)

        curr_graph = Graph(x=x_node, edge_index=raw_edge_index, x_reid=x_reid,y_id=y_id,  
                            fwrd_vel=fwrd_vel, bwrd_vel=bwrd_vel, 
                            x_frame_start=x_frame[0], x_frame_end=x_frame[1],
                            x_center_start = torch.cat((x_center[0], x_bbox[0][:, 2:]), dim=1),
                            x_center_end = torch.cat((x_center[1], x_bbox[1][:, 2:]), dim=1),
                            x_box_start=x_bbox[0], x_box_end=x_bbox[1],
                            x_frame_mask=x_frame_mask, x_feet_start = x_feet[0], x_feet_end=x_feet[1],
                            pruning_score=torch.zeros(raw_edge_index.shape[1], device=x_node.device), # Placeholder needed for correct batching
                            **motion_features)

        return curr_graph

    def add_edges_to_curr_graph(self, config, curr_graph):
        """
        Prune edges from a set of time-valid edge connections and compute edge features for them
        """ 
        x_frame = (curr_graph.x_frame_start, curr_graph.x_frame_end)
        x_feet = (curr_graph.x_feet_start, curr_graph.x_feet_end)
        x_frame_mask = curr_graph.x_frame_mask if hasattr(curr_graph, 'x_frame_mask') else None
        x_reid = curr_graph.x_reid
        x_bbox = (curr_graph.x_box_start, curr_graph.x_box_end)
        
        # Prune time-valid edges
        reid_sim_fn = F.pairwise_distance if config.reid_sim_fn == 'l2' else lambda x, y: 2 - F.cosine_similarity(x, y)
        edge_ixs, edge_mask=prune_edges(edge_ixs=curr_graph.edge_index, node_frames=x_frame, node_reids=x_reid, top_k_nns=config.top_k_nns, 
                                        depth=self.curr_depth, node_feets=x_feet, graph_pruning_score=curr_graph.pruning_score, pruning_method = config.pruning_method,
                                        reid_sim_fn=reid_sim_fn)
        if curr_graph.pruning_score is not None:
            curr_graph.pruning_score = curr_graph.pruning_score[edge_mask]
        
        # Edge features
        motion_feats = {'motion_giou': curr_graph.pruning_score}
        
        edge_feats_to_use_ = ['secs_time_dists']
        if config.mpn_use_pos_edge[self.curr_depth]:
            edge_feats_to_use_ += ['norm_feet_x_dists', 'norm_feet_y_dists', 'bb_height_dists', 'bb_width_dists']
        
        if config.mpn_use_reid_edge[self.curr_depth]:
            edge_feats_to_use_ += ['emb_dists']

        if config.mpn_use_motion[self.curr_depth]:
            edge_feats_to_use_ += ['motion_giou']
                        
        reid_sim_fn = F.pairwise_distance if config.edge_sim_fn == 'l2' else lambda x, y: 2 - F.cosine_similarity(x, y)

        edge_features = compute_edge_features(edge_ixs=edge_ixs, node_frames=x_frame, node_bboxes=x_bbox,
                                              node_feets=x_feet, node_reids=x_reid, fps=self.fps,
                                              edge_feats_to_use = edge_feats_to_use_,
                                              motion_feats=motion_feats,
                                              reid_sim_fn=reid_sim_fn)

        if config.symmetric_edges:
            edge_ixs, edge_features = make_symmetric_edges(edge_ixs=edge_ixs, edge_features=edge_features)

        # LABELS - if train/val
        edge_labels, edge_mask = assign_edge_labels(node_ids=curr_graph.y_id, edge_ixs=edge_ixs, node_frames=x_frame)

        # Placeholder for edge predictions
        edge_preds = torch.zeros((edge_features.shape[0]), dtype=edge_features.dtype)

        # Construct the current layer graph
        curr_graph = Graph(x=curr_graph.x, edge_attr=edge_features, edge_index=edge_ixs, edge_labels=edge_labels,
                               y_id=curr_graph.y_id,
                           edge_preds=edge_preds, x_reid=x_reid, edge_mask=edge_mask)

        return curr_graph

    def update_maps_and_depth(self, labels):
        """
        After node ids are assigned, this function updates the maps of the hierarchical graph

        labels: Id for each of the last layer nodes. Each id will represent a new node in the next layer
        """

        labels = torch.tensor(labels).long().to(self.device())

        # Update the map_from_init
        if self.curr_depth == 0:
            self.map_from_init = labels
        else:
            self.map_from_init = labels[self.map_from_init]

        # Update all maps and current depth
        self.maps.append(labels)
        self.curr_depth += 1

    def update_maps_and_depth_wo_labels(self):
        """
        In case no labels are predicted, use this function for update. E.g: No edges were found in the graph.
        With this function, map_from_init remains the same
        """

        self.maps.append([])
        self.curr_depth += 1

    def get_labels(self):
        return self.map_from_init.detach().cpu().numpy()

    def _change_attrs_types(self, attr_change_fn):
        """
        Change attribute types with a given function
        """
        # These are our standard 'data-related' attribute names.
        _data_attr_names = [
                            # Graph related
                            'x_node',  # Node features that will be processed
                            'x_reid',  # ReID features to calculate edge connections
                            'x_frame',  # Frame number of the detection
                            'x_one_hot_frame',  # Frame numbers - one hot encoded
                            'x_bbox',  # Bounding box coordinates of the node (left, top, W, H)
                            'x_center',  # Bonuding box center coordinates of the node
                            'x_feet',  # Further bbox coordinates (feet_x, feet_y)
                            'y_id',  # GT identity of the node

                            # Scene related
                            'fps',  # FPS of the sequence
                            'frames_total',  # Total number of frames
                            'frames_per_level',  # Frame number per level
                            'start_frame',  # Start frame of a graph
                            'end_frame',  # End frame of a graph

                            # Hierarchy related
                            'map_from_init',
                            'curr_depth'
                            ]

        for attr_name in _data_attr_names:
            if hasattr(self, attr_name):
                if getattr(self, attr_name ) is not None:
                    old_attr_val = getattr(self, attr_name)
                    setattr(self, attr_name, attr_change_fn(old_attr_val))

    def tensor(self):
        self._change_attrs_types(attr_change_fn= torch.tensor)
        return self

    def float(self):
        self._change_attrs_types(attr_change_fn= lambda x: x.float())
        return self

    def numpy(self):
        self._change_attrs_types(attr_change_fn= lambda x: x if isinstance(x, np.ndarray) else x.detach().cpu().numpy())
        return self

    def cpu(self):
        self._change_attrs_types(attr_change_fn= lambda x: x.cpu())
        return self

    def cuda(self):
        self._change_attrs_types(attr_change_fn=lambda x: x.cuda())
        return self

    def to(self, device):
        self._change_attrs_types(attr_change_fn=lambda x: x.to(device))

    def device(self):
        if isinstance(self.x_node, torch.Tensor):
            return self.x_node.device
        return torch.device('cpu')
