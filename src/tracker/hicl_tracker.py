from pickletools import optimize
import yaml
import torch.optim as optim
from torch_geometric.data import Batch
from src.models.mpntrack import MOTMPNet
from src.models.hiclnet import HICLNet
from src.models.motion.linear import LinearMotionModel
from src.utils.deterministic import seed_worker, seed_generator
from src.data.mot_datasets import MOTSceneDataset
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch
from src.utils.graph_utils import to_undirected_graph, to_lightweight_graph, to_positive_decision_graph
from src.utils.motion_utils import compute_giou_fwrd_bwrd_motion_sim
from src.tracker.projectors import GreedyProjector, ExactProjector
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
import pandas as pd
from lapsolver import solve_dense
import os.path as osp
from src.tracker.postprocessing import Postprocessor
import time
import statistics
import os
from TrackEval.scripts.run_mot_challenge import evaluate_mot17
import matplotlib.pyplot as plt
from torch import nn
import math
import rama_py
import pickle
from torch.utils.tensorboard.writer import SummaryWriter

class HICLTracker:
    """
    Hierarchical processor of the sequences. Consists of a trainable model and hierarchical processing scripts
    """
    def __init__(self, config, seqs, splits):
        self.config = config
        self.seqs = seqs
        self.train_split, self.val_split, self.test_split = splits

        # Load the model (currently MPNTrack)
        self.model = self._get_model()

        if self.config.do_motion:
            self.motion_model = LinearMotionModel()
        
        else:
            self.motion_model = None

        # Training - Set up the dataset and optimizer
        if self.config.experiment_mode in ('train', 'train-cval'):
            self.loss_function = FocalLoss(logits=True, gamma=self.config.gamma)
            self.train_dataloader = self._get_train_dataloader()
            self.optimizer = self._get_optimizer()
            # Get validation dataset if exists
            if self.val_split:
                self.val_dataset = self._get_dataset(mode='val')

        # Testing - Set up the dataset
        elif self.config.experiment_mode == 'test':
            self.test_dataset = self._get_dataset(mode='test')

        # Iteration and epoch
        self.train_iteration = 0
        self.train_epoch = 0

        # Layers that are allowed to be trained
        if self.config.depth_pretrain_iteration == 0:
            self.active_train_depth = self.config.hicl_depth
        else:
            self.active_train_depth = min(1, self.config.hicl_depth)

        #if self.config.tensorboard:
        if self.config.experiment_mode == 'train':
            self.logger = SummaryWriter(osp.join(self.config.experiment_path, 'tf_logs'))

    def _get_model(self):
        """
        Load the hierarchical model
        """

        # Read mpntrack config file
        with open(r'configs/mpntrack_cfg.yaml') as file:
            mpntrack_params = yaml.load(file, Loader=yaml.FullLoader)


        mpntrack_params['graph_model_params']['encoder_feats_dict']['node_in_dim'] = self.config.node_dim

        # Update the HICL feats config
        mpntrack_params['graph_model_params']['do_hicl_feats']=self.config.do_hicl_feats
        mpntrack_params['graph_model_params']['hicl_feats_encoder'].update(self.config.hicl_feats_args)


        return HICLNet(submodel_type=MOTMPNet, submodel_params=mpntrack_params['graph_model_params'],
                       hicl_depth=self.config.hicl_depth, use_motion=self.config.mpn_use_motion,
                       use_reid_edge=self.config.mpn_use_reid_edge, use_pos_edge=self.config.mpn_use_pos_edge,
                       share_weights=self.config.share_weights, edge_level_embed=self.config.edge_level_embed,
                       node_level_embed=self.config.node_level_embed
                       ).to(self.config.device)

    def _get_dataset(self, mode):
        """
        Create dataset objects
        """
        return MOTSceneDataset(config=self.config, seqs=self.seqs[mode], mode=mode)

    def _get_train_dataloader(self):
        """
         Set up the dataset and the dataloader
        """

        train_dataset = self._get_dataset(mode='train')
        train_loader = DataLoader(train_dataset, batch_size=self.config.num_batch, num_workers=self.config.num_workers,
                                  shuffle=True,
                                  worker_init_fn=seed_worker, generator=seed_generator(), )
        return train_loader

    def _get_optimizer(self):
        """
         Set up the optimizer
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        return optimizer

    def _get_projector(self, graph):
        """
        Set up the projector that will round the edge predictions
        """
        if self.config.rounding_method == 'greedy':
            projector = GreedyProjector(full_graph=graph)

        elif self.config.rounding_method == 'exact':
            projector = ExactProjector(full_graph=graph, solver_backend=self.config.solver_backend)

        else:
            raise RuntimeError("Rounding type for projector not understood")
        return projector

    def _save_model(self):
        """
        Save the model
        """
        # Create models folder
        model_path = osp.join(self.config.experiment_path, "models")
        os.makedirs(model_path, exist_ok=True)

        # Create the file
        file_name = osp.join(model_path, f"hiclnet_epoch_{self.train_epoch}_iteration{self.train_iteration}.pth")
        torch.save(self.model.state_dict(), file_name)

        # Create a full checkpoint
        checkpoint_path = osp.join(self.config.experiment_path, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
        cp_file_name = osp.join(checkpoint_path, f"checkpoint_{self.train_epoch}.pth")
        torch.save({'epoch': self.train_epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, cp_file_name)

    def load_pretrained_model(self):
        """
        Load a pretrained model
        """
        # Initialize the model
        pretrained_model = self._get_model()

        # Load weights
        pretrained_model.load_state_dict(torch.load(self.config.hicl_model_path))
        pretrained_model.train()  # Put in training mode for now

        return pretrained_model

    def _hicl_to_curr(self, hicl_graphs):
        """
        Method that creates a batch of current graphs from hierarchical graphs in three steps:
        1) Create the batch graphs with node features and all time valid edge connections
        2) **Optionally** compute motion features for all time valid edge connections
        3) Use those motion features, as well as reid to define KNN edges and define 
        edge features for each graph in the batch to obtain the final graphs
        """
        batch = Batch.from_data_list([hicl_graph.construct_curr_graph_nodes(self.config)
                                                        for hicl_graph in hicl_graphs])
        
        curr_depth = hicl_graphs[0].curr_depth
        if self.config.do_motion and curr_depth >0:
            motion_pred = self.predict_motion(batch, curr_depth = curr_depth)
            batch.pruning_score = compute_giou_fwrd_bwrd_motion_sim(batch, motion_pred)
            
            if 'estimate_vel' in motion_pred[0]:
                batch.fwrd_vel, batch.bwrd_vel = motion_pred[0]['estimate_vel'], motion_pred[1]['estimate_vel']
                            
        else:
            motion_pred = None
        
        # Now unbatch graphs, add their remaining features, and batch them again
        curr_graphs = Batch.to_data_list(batch)
        curr_graph_batch = Batch.from_data_list([hicl_graph.add_edges_to_curr_graph(self.config, curr_graph)
                                                 for curr_graph, hicl_graph in zip(curr_graphs, hicl_graphs) if ((curr_graph.edge_index is not None) and (curr_graph.edge_index.numel()))])

        return curr_graph_batch, motion_pred

    def _postprocess_graph(self, graph, remove_negatives=True, decision_threshold=0.5):
        """
        Process the graph before feeding it to the projector
        """
        to_undirected_graph(graph, attrs_to_update=('edge_preds', 'edge_labels'))
        to_lightweight_graph(graph, attrs_to_del=('reid_emb_dists', 'x', 'edge_attr', 'edge_labels'))
        if remove_negatives:
            to_positive_decision_graph(graph, decision_threshold)

    def _project_graph(self, graph, decision_threshold=0.5):
        """
        Project model output with a solver
        """
        projector = self._get_projector(graph=graph)
        projector.project(decision_threshold)
        graph = graph.numpy()
        graph.constr_satisf_rate = projector.constr_satisf_rate
        return graph

    def _assign_labels(self, graph):
        """
        Clusters the nodes together based on edge predictions
        """
        # Only keep the non-zero edges
        nonzero_mask = graph.edge_preds == 1
        nonzero_edge_index = graph.edge_index.T[nonzero_mask].T
        nonzero_edges = graph.edge_preds[nonzero_mask].astype(int)
        graph_shape = (graph.num_nodes, graph.num_nodes)

        # Express the result as a CSR matrix so that it can be fed to 'connected_components')
        csr_graph = csr_matrix((nonzero_edges, (tuple(nonzero_edge_index))), shape=graph_shape)

        # Get the connected Components:
        n_components, labels = connected_components(csgraph=csr_graph, directed=False, return_labels=True)

        return n_components, labels

    def _calculate_loss(self, outputs, edge_labels, edge_mask):
        """
        Calculate MPNTrack loss given edge predictions and edge_labels
        """

        # Compute Weighted BCE:
        loss = torch.as_tensor([.0], device=self.config.device)
        num_steps = len(outputs['classified_edges'])
        for step in range(num_steps):
            if self.config.no_fp_loss and torch.any(edge_mask).item():
                loss += self.loss_function(outputs['classified_edges'][step].view(-1)[edge_mask], edge_labels.view(-1)[edge_mask])
            else:
                loss += self.loss_function(outputs['classified_edges'][step].view(-1), edge_labels.view(-1))
        return loss

    def _train_epoch(self):
        """
        Train a single epoch
        """
        logs = {"Loss": [], "Loss_per_Depth": [[] for j in range(self.config.hicl_depth)], "Time": []}

        for i, train_batch in enumerate(self.train_dataloader):
            t_start = time.time()

            # Iteration update
            self.train_iteration += 1

            self.optimizer.zero_grad()

            train_batch.to(self.config.device)  # Send batch to the device
            hicl_graphs = train_batch.to_data_list()  # Initialize the hierarchical graphs

            loss = torch.as_tensor([.0], device=self.config.device)  # Initialize the batch loss

            _, loss, logs = self.hicl_forward(hicl_graphs = hicl_graphs, 
                                              logs = logs, 
                                              oracle = False, 
                                              mode = 'train', 
                                              max_depth = self.active_train_depth, 
                                              project_max_depth = self.active_train_depth - 1)

            # Update the weights
            loss.backward()
            self.optimizer.step()

            # Keep track of the logs
            t_end = time.time()
            logs["Loss"].append(loss.detach().item())
            logs["Time"].append(t_end-t_start)

            self._log_tb_train_metrics(logs)

            # Verbose
            if i % self.config.verbose_iteration == 0 and i != 0:
                print(f"Iteration {i} / {len(self.train_dataloader)} - Training Loss:", statistics.mean(logs["Loss"][i-self.config.verbose_iteration:i]), '- Time:', sum(logs["Time"][i-self.config.verbose_iteration:i]))  # Verbose

            # Update active train depth if required
            if self.active_train_depth < self.config.hicl_depth:
                if self.train_iteration % self.config.depth_pretrain_iteration == 0:
                    self.active_train_depth = min(self.active_train_depth+1, self.config.hicl_depth)
                    print("*****")
                    print("Frozen layers are unlocked! Current active training depth is:", self.active_train_depth)
                    print("*****")

        self.train_epoch += 1

        return logs

    def predict_motion(self, batch, curr_depth):
        """Predict forward and backward future/past locations for each track with length >1"""

        motion_model = self.motion_model
        assert motion_model is not None

        fwrd_motion_pred = motion_model(x_motion=batch.x_fwrd_motion, 
                                        x_last_pos=batch.x_center_end[~batch.x_ignore_traj],
                                        pred_length=self.config.motion_pred_length[curr_depth - 1],
                                        linear_center_only=self.config.linear_center_only
                                        )
        
        bwrd_motion_pred = motion_model(x_motion=batch.x_bwrd_motion, 
                                        x_last_pos=batch.x_center_start[~batch.x_ignore_traj],
                                        pred_length=self.config.motion_pred_length[curr_depth - 1],
                                        linear_center_only=self.config.linear_center_only
                                        )            

        return (fwrd_motion_pred, bwrd_motion_pred)

    def hicl_forward(self, hicl_graphs, logs, oracle, mode, max_depth, project_max_depth):
        hicl_feats=None
        loss = torch.as_tensor([.0], device=self.config.device)  # Initialize the batch loss
        
        # For each depth
        for curr_depth in range(max_depth):
        
            # Put the graph into the correct format
            curr_batch, _ = self._hicl_to_curr(hicl_graphs=hicl_graphs)  # Create curr_graphs from hierarachical graphs            
            batch_idx = curr_batch.batch

            if curr_depth == 0 or not self.config.do_hicl_feats:
                curr_batch.hicl_feats = None

            elif hicl_feats is not None:
                curr_batch.hicl_feats = hicl_feats

            # Forward pass if there is an edge
            if curr_batch.edge_index.numel():                
                if oracle:
                    # Oracle results
                    curr_batch.edge_preds = curr_batch.edge_labels
                    logs[curr_depth]["Loss"].append(0.)
                    # Calculate batch classification metrics and loss
                    logs[curr_depth] = self._calculate_true_false_metrics(edge_preds=curr_batch.edge_preds,
                                                                edge_labels=curr_batch.edge_labels, 
                                                                logs=logs[curr_depth])
                else:
                    # Graph based forward pass
                    outputs = self.model(curr_batch, curr_depth)  # Forward pass for this specific depth
                    
                    # Produce decisions
                    curr_batch.edge_preds = torch.sigmoid(outputs['classified_edges'][-1].view(-1).detach())

                    if mode == 'val':
                        # Calculate the batch loss
                        logs[curr_depth]["Loss"].append(self._calculate_loss(outputs=outputs, edge_labels=curr_batch.edge_labels, edge_mask=curr_batch.edge_mask).item())
                        # Calculate batch classification metrics and loss
                        logs[curr_depth] = self._calculate_true_false_metrics(edge_preds=curr_batch.edge_preds,
                                                                    edge_labels=curr_batch.edge_labels, logs=logs[curr_depth])
                    elif mode == 'train':
                        # Calculate loss and prepare for a forward pass
                        loss_curr_depth = self._calculate_loss(outputs=outputs, edge_labels=curr_batch.edge_labels, edge_mask=curr_batch.edge_mask)                          
                        loss += loss_curr_depth
                        
                        logs["Loss_per_Depth"][curr_depth].append(loss_curr_depth.detach().item())  # log the curr loss


            graph_data_list = curr_batch.to_data_list()
            if mode != 'train':
                assert len(graph_data_list) == 1, "Track batch size is greater than 1"

            hicl_feats = []
            if curr_depth < project_max_depth:  # Last layer update is not necessary for training
                for ix_graph, graph in enumerate(graph_data_list):        
                    if graph.edge_index.numel():
                        # Flow formulation
                        if self.config.formulation == 'flow':
                            # Process the graph before feeding it to the projector
                            self._postprocess_graph(graph)
                            # Project model output with a solver
                            graph = self._project_graph(graph)
                            
                            # Assign ped ids
                            n_components, labels = self._assign_labels(graph)
                        
                        node_mask = batch_idx == ix_graph
                        if self.config.do_hicl_feats and not oracle:
                            hicl_feats.append(self.model.layers[curr_depth].hicl_feats_encoder.pool_node_feats(outputs['node_feats'][node_mask], labels))

                        # Update the hierarchical graphs with new map_from_init and depth
                        hicl_graphs[ix_graph].update_maps_and_depth(labels)

                    else:
                        # Update the hierarchical graphs
                        hicl_graphs[ix_graph].update_maps_and_depth_wo_labels()
            
            if len(hicl_feats) > 0:
                hicl_feats = torch.cat(hicl_feats)
            
            else:
                hicl_feats =None

        return hicl_graphs, loss, logs

    def _log_tb_class_metrics(self, epoch_val_logs, epoc_val_logs_per_depth):
        # Struct is 'layer name :  metric_name :val
        if self.logger is not None:
            _METRICS_TO_LOG = ['Loss','Precision', 'Recall', 'F1', 'Accuracy']
            prefixs = [f'layer_{layer_idx + 1}' for layer_idx in range(len(epoc_val_logs_per_depth))] + ['overall']
            tb_logs = epoc_val_logs_per_depth + [epoch_val_logs]
            for prefix, logs_dict in zip(prefixs, tb_logs):
                for metric_name, metric_val in logs_dict.items():
                    if metric_name in _METRICS_TO_LOG:
                        tag = '/'.join(['val', 'secondary', prefix, metric_name])
                        self.logger.add_scalar(tag, metric_val, global_step=self.train_iteration)

    def _log_tb_train_metrics(self, logs):
        if self.logger is not None:
            loss_logs = logs['Loss_per_Depth'] + [logs['Loss']]
            prefixes = [f'layer_{layer_idx + 1}' for layer_idx in range(len(logs['Loss_per_Depth']))] + ['overall']

            for prefix, losses in zip(prefixes, loss_logs):
                if len(losses) > 0:
                    tag = '/'.join(['train', prefix, 'loss'])
                    self.logger.add_scalar(tag, losses[-1], global_step=self.train_iteration)

    def _log_tb_mot_metrics(self, mot_metrics):
        if self.logger is not None:
            path = list(mot_metrics['MotChallenge2DBox'].keys())[0]
            _METRICS_GROUPS = ['HOTA', 'CLEAR', 'Identity']
            _METRICS_TO_LOG = ['HOTA','AssA', 'DetA', 'MOTA', 'IDF1']

            metrics_ = mot_metrics['MotChallenge2DBox'][path]['COMBINED_SEQ']['pedestrian'] # We may need more classes for MOTCha
            for metrics_group_name in _METRICS_GROUPS:
                group_metrics = metrics_[metrics_group_name]
                for metric_name, metric_val in group_metrics.items():
                    if metric_name in _METRICS_TO_LOG:
                        if isinstance(metric_val, np.ndarray):
                            metric_val = np.mean(metric_val)
                        tag = '/'.join(['val', 'mot', metric_name])
                        self.logger.add_scalar(tag, metric_val, global_step=self.train_iteration)


    def train(self):
        """
        Perform a full training
        """

        # First calculate the oracle results for validation
        if self.val_split:
        #if False:
            _, _ = self.track(self.val_dataset, output_path=osp.join(self.config.experiment_path, 'oracle'), mode='val', oracle=True)
            evaluate_mot17(tracker_path=osp.join(self.config.experiment_path, 'oracle'), split=self.val_split,
                           data_path=self.config.data_path, tracker_sub_folder=self.config.mot_sub_folder,
                           output_sub_folder=self.config.mot_sub_folder)
        #raise RuntimeError
        assert self.model.training, "Training error: Model is not in training mode"

        logs = {"Train_Loss": [], "Val_Loss": [],
                "Train_Loss_per_Depth": [[] for j in range(self.config.hicl_depth)],
                "Val_Loss_per_Depth": [[] for j in range(self.config.hicl_depth)]}  # Training logs

        # Training loop
        for epoch in range(1, self.config.num_epoch+1):
            t_start = time.time()
            print("###############")
            print("     Epoch ", epoch)
            print("###############")

            # Create epoch output dir
            epoch_path = osp.join(self.config.experiment_path, 'Epoch' + str(epoch))
            os.makedirs(epoch_path, exist_ok=True)

            # Train for one epoch
            epoch_train_logs = self._train_epoch()

            # Train loss logs
            logs["Train_Loss"].append(statistics.mean(epoch_train_logs["Loss"]))
            for j in range(self.config.hicl_depth):
                # If a layer is frozen, set the loss to 0
                if j < self.active_train_depth and len(epoch_train_logs["Loss_per_Depth"][j]) > 0:
                    logs["Train_Loss_per_Depth"][j].append(statistics.mean(epoch_train_logs["Loss_per_Depth"][j]))
                else:
                    logs["Train_Loss_per_Depth"][j].append(0.)

            # Validation steps
            if self.val_split and epoch >= self.config.start_eval:
                epoch_val_logs, epoc_val_logs_per_depth = self.track(dataset=self.val_dataset, output_path=epoch_path, mode='val', oracle=False)
                # Validation logs
                logs["Val_Loss"].append(epoch_val_logs["Loss"])
                for j in range(self.config.hicl_depth):
                    logs["Val_Loss_per_Depth"][j].append(epoc_val_logs_per_depth[j]["Loss"])

                # Tensorboard logging:
                self._log_tb_class_metrics(epoch_val_logs, epoc_val_logs_per_depth)

                # MOT metrics
                mot_metrics= evaluate_mot17(tracker_path=epoch_path, split=self.val_split, data_path=self.config.data_path,
                                            tracker_sub_folder=self.config.mot_sub_folder, output_sub_folder=self.config.mot_sub_folder)[0]
                self._log_tb_mot_metrics(mot_metrics) 

            # Plot losses
            self._plot_losses(logs)
            self._plot_losses_per_depth(logs)

            # Save model checkpoint
            if self.config.save_cp:
                self._save_model()

            # Time information
            t_end = time.time()
            print(f"Epoch completed in {round((t_end - t_start) / 60, 2)} minutes")

    def track(self, dataset, output_path, mode='val', oracle=False):
        """
        Main tracking method. Given a dataset, track every sequence in it and create output files.
        """

        # Set the model in the test mode
        self.model.eval()
        assert not self.model.training, "Test error: Model is not in evaluation mode"

        # Disable gradients
        with torch.no_grad():

            # Separate dictionary to bookkeep the logs for each depth network
            logs_all = [{"Loss": [], "TP": [], "FP": [], "TN": [], "FN": [], "CSR": []} for i in range(self.config.hicl_depth)]

            # Loop over sequences
            for seq, seq_and_frames in dataset.sparse_frames_per_seq.items():
                print("Tracking", seq)
                # Loop over each datapoint - Equivalent to using a dataloader with batch 1
                seq_dfs = []
                for seq_name, start_frame, end_frame in seq_and_frames:
                    # Equivalent to train_batch with a single datapoint
                    data = dataset.get_graph_from_seq_and_frames(seq_name=seq_name, start_frame=start_frame, end_frame=end_frame)
                    data.to(self.config.device)

                    # Small trick to utilize torch-geometric built in functions
                    track_batch = Batch.from_data_list([data])
                    hicl_graphs = track_batch.to_data_list()

                    # For each depth
                    hicl_graphs, _, logs_all = self.hicl_forward(hicl_graphs = hicl_graphs, 
                                                                 logs = logs_all, 
                                                                 oracle = oracle, 
                                                                 mode = mode, 
                                                                 max_depth = self.config.hicl_depth, 
                                                                 project_max_depth = self.config.hicl_depth)

                    # Pedestrian ids
                    ped_labels = hicl_graphs[0].get_labels()

                    # Get graph df
                    graph_df, _ = dataset.get_df_from_seq_and_frames(seq_name=seq_name, start_frame=start_frame, end_frame=end_frame)
                    assert len(ped_labels) == graph_df.shape[0], "Ped Ids Label format is wrong"

                    # Each Connected Component is a Ped Id. Assign those values to our DataFrame:
                    graph_output_df = graph_df.copy()
                    graph_output_df['ped_id'] = ped_labels
                    graph_output_df = graph_output_df[self.config.VIDEO_COLUMNS + ['conf', 'detection_id']].copy()

                    # Append the new df
                    seq_dfs.append(graph_output_df)

                # Merge the dataframes
                seq_merged_df = self._merge_subseq_dfs(seq_dfs)

                # Postprocess the dataframes - drop short trajectories
                postprocess = Postprocessor(seq_merged_df.copy(),
                                            seq_info_dict=dataset.seq_info_dicts[seq_name],
                                            config=self.config)
                seq_output_df = postprocess.postprocess_trajectories()

                # Save the output
                os.makedirs(osp.join(output_path, self.config.mot_sub_folder), exist_ok=True)
                tracking_file_path = osp.join(output_path, self.config.mot_sub_folder, seq_name + '.txt')
                self._save_results(df=seq_output_df, output_file_path=tracking_file_path)

            print("-----")
            print("Tracking completed!")

            # Print metrics
            logs_total = {}
            if mode != 'test':
                for i in range(self.config.hicl_depth):
                    # Calculate accuracy, precision, recall
                    logs = logs_all[i]
                    if logs["Loss"]:
                        print("Depth", i+1, "- Metrics:")
                        logs = self._postprocess_logs(logs=logs)

                # Total logs - Cumulative of every layer
                print("TOTAL - Metrics:")
                logs_total = {"Loss": [logs_all[i]["Loss"] for i in range(self.config.hicl_depth)],
                              "TP": [logs_all[i]["TP"] for i in range(self.config.hicl_depth)],
                              "FP": [logs_all[i]["FP"] for i in range(self.config.hicl_depth)],
                              "TN": [logs_all[i]["TN"] for i in range(self.config.hicl_depth)],
                              "FN": [logs_all[i]["FN"] for i in range(self.config.hicl_depth)],
                              "CSR": [logs_all[i]["CSR"] for i in range(self.config.hicl_depth)]}
                logs_total["Loss"] = [sum(logs_total["Loss"])]  # To be compatible with train loss (sum of hicl layers)
                logs_total = self._postprocess_logs(logs_total)

                print("-----")



        # Set the model back in training mode
        self.model.train()

        return logs_total, logs_all

    def _merge_subseq_dfs(self, subseq_dfs):
        """
        Experimental merge dfs copied from MPNTrack. Might not be optimized. Check its behavior.

            Algorithm:
            Create a df consisting of ids from the left, ids from the right and how many times they match with
            each other based on the detection id.

            Create a cost matrix that has NaN everywhere except the matched ids. The cost at these locations are
            -(num_matched). E.g: if id 1 and 100 matched 10 times -> cost_matrix[1, 100] = -10

            Solve the cost matrix and find the assignments

            Merge both dataframes and replace the larger of the matched ids with the smaller one

        """
        seq_df = subseq_dfs[0]
        for subseq_df in subseq_dfs[1:]:
            # Make sure that ped_ids in subseq_df are new and all greater than the ones in seq_df:
            subseq_df['ped_id'] += seq_df['ped_id'].max() + 1

            intersect_frames = np.intersect1d(seq_df.frame, subseq_df.frame)  # Common frames between 2 dfs

            # Detections in common frames within seq_df
            left_df = seq_df[['detection_id', 'ped_id']][seq_df.frame.isin(intersect_frames)]
            left_ids_pos = left_df[['ped_id']].drop_duplicates();
            left_ids_pos['ped_id_pos'] = np.arange(left_ids_pos.shape[0])
            left_df = left_df.merge(left_ids_pos, on='ped_id').set_index('detection_id')

            # Detections in common frames within subseq_df
            right_df = subseq_df[['detection_id', 'ped_id']][subseq_df.frame.isin(intersect_frames)]
            right_ids_pos = right_df[['ped_id']].drop_duplicates();
            right_ids_pos['ped_id_pos'] = np.arange(right_ids_pos.shape[0])
            right_df = right_df.merge(right_ids_pos, on='ped_id').set_index('detection_id')

            # Count how many times each left_id corresponds to right_id (based on detection_id)
            common_boxes = \
                left_df[['ped_id_pos']].join(right_df['ped_id_pos'], lsuffix='_left', rsuffix='_right').dropna(
                    thresh=2).reset_index().groupby(['ped_id_pos_left', 'ped_id_pos_right'])['detection_id'].count()
            common_boxes = common_boxes.reset_index().astype(int)

            # Create a cost matrix with negative count (more match, less cost). Everywhere else is NaN
            cost_mat = np.full((common_boxes['ped_id_pos_left'].max() + 1, common_boxes['ped_id_pos_right'].max() + 1),
                               fill_value=np.nan)
            cost_mat[common_boxes['ped_id_pos_left'].values, common_boxes['ped_id_pos_right'].values] = - common_boxes[
                'detection_id'].values

            # Find the min cost solution
            matched_left_ids_pos, matched_right_ids_pos = solve_dense(cost_mat)

            # Map of the matched ids
            matched_ids = pd.DataFrame(data=np.stack((left_ids_pos['ped_id'].values[matched_left_ids_pos],
                                                      right_ids_pos['ped_id'].values[matched_right_ids_pos])).T,
                                       columns=['left_ped_id', 'right_ped_id'])

            # Assign the ids matched to subseq_df
            subseq_df = pd.merge(subseq_df, matched_ids, how='outer', left_on='ped_id', right_on='right_ped_id')
            subseq_df['left_ped_id'].fillna(np.inf, inplace=True)
            subseq_df['ped_id'] = np.minimum(subseq_df['left_ped_id'], subseq_df['ped_id'])

            # Update seq_df
            seq_df = pd.concat([seq_df, subseq_df[subseq_df['frame'] > seq_df['frame'].max()]])
        return seq_df.reset_index(drop=True)

    def _calculate_true_false_metrics(self, edge_preds, edge_labels, logs):
        """
        Calculate TP, FP, TN, FN
        """

        # edge_preds needs to be already after a sigmoid
        preds = (edge_preds.view(-1) > 0.5).float()

        # Metrics
        TP = ((edge_labels == 1) & (preds == 1)).sum().float()
        FP = ((edge_labels == 0) & (preds == 1)).sum().float()
        TN = ((edge_labels == 0) & (preds == 0)).sum().float()
        FN = ((edge_labels == 1) & (preds == 0)).sum().float()

        # Update the logs
        logs["TP"].append(TP.item())
        logs["FP"].append(FP.item())
        logs["TN"].append(TN.item())
        logs["FN"].append(FN.item())

        return logs

    def _postprocess_logs(self, logs):
        """
        Calculate accuracy, precision, recall
        """
        logs["Loss"] = statistics.mean(logs["Loss"])
        logs["TP"] = sum(logs["TP"])
        logs["FP"] = sum(logs["FP"])
        logs["TN"] = sum(logs["TN"])
        logs["FN"] = sum(logs["FN"])

        logs["Accuracy"] = (logs["TP"] + logs["TN"]) / (logs["TP"] + logs["FP"] + logs["TN"] + logs["FN"])
        logs["Recall"] = logs["TP"] / (logs["TP"] + logs["FN"]) if logs["TP"] + logs["FN"] > 0 else 0
        logs["Precision"] = logs["TP"] / (logs["TP"] + logs["FP"]) if logs["TP"] + logs["FP"] > 0 else 0
        logs["F1"] = 2*logs["TP"] / (2*logs["TP"] + logs['FP']+ logs['FN']) if (logs["TP"] + logs["FP"] +logs['FN']) > 0 else 0

        if logs["CSR"]:
            logs["CSR"] = statistics.mean(logs["CSR"])
        else:
            logs["CSR"] = math.nan

        # Verbose
        print("     Loss: ", logs["Loss"])
        print("     Accuracy: ", logs["Accuracy"])
        print("     Recall: ", logs["Recall"])
        print("     Precision: ", logs["Precision"])
        print("     Constraint Satisfaction Rate: ", logs["CSR"])
        print("     TP+FP+TN+FN: ", int((logs["TP"] + logs["FP"] + logs["TN"] + logs["FN"])))

        return logs

    def _plot_losses(self, logs):
        """
        Plot training and validation losses
        """
        plt.figure(figsize=(12, 9))
        plt.plot(logs["Train_Loss"], label='Training Loss')

        if logs["Val_Loss"]:
            plt.plot(logs["Val_Loss"], label='Validation Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(len(logs["Train_Loss"])), np.arange(1, len(logs["Train_Loss"]) + 1))
        plt.savefig(osp.join(self.config.experiment_path, 'loss_plots-' + self.train_split + '.png'), bbox_inches='tight')
        plt.close()

    def _plot_losses_per_depth(self, logs):
        """
        Plot training and validation losses per depth
        """
        plt.figure(figsize=(12, 9))
        for j in range(self.config.hicl_depth):
            plt.plot(logs["Train_Loss_per_Depth"][j], label=f'Training Loss - Depth {j}')
            if logs["Val_Loss"]:
                plt.plot(logs["Val_Loss_per_Depth"][j], label=f'Validation Loss - Depth {j}')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(len(logs["Train_Loss"])), np.arange(1, len(logs["Train_Loss"]) + 1))
        plt.savefig(osp.join(self.config.experiment_path, 'loss_plots_per_depth-' + self.train_split + '.png'), bbox_inches='tight')
        plt.close()

    def _plot_metrics(self, logs):
        """
        Plot validation metrics
        """
        plt.figure(figsize=(12, 9))
        plt.plot(logs["Val_HOTA"], label='Validation HOTA')
        plt.plot(logs["Val_MOTA"], label='Validation MOTA')
        plt.plot(logs["Val_IDF1"], label='Validation IDF1')

        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(len(logs["Train_Loss"])), np.arange(1, len(logs["Val_HOTA"]) + 1))

        plt.savefig(osp.join(self.config.experiment_path, 'val_metrics-' + self.train_split + '-' + self.val_split + '.png'), bbox_inches='tight')

    def _save_results(self, df, output_file_path):
        """
        Save the tracking output df
        """
        df['conf'] = 1
        df['x'] = -1
        df['y'] = -1
        df['z'] = -1
        df['ped_id'] = df['ped_id'].astype('int64')

        # Coordinates are 1 based - revert
        df['bb_left'] += 1
        df['bb_top'] += 1

        final_out = df[self.config.TRACKING_OUT_COLS].sort_values(by=['frame', 'ped_id'])
        final_out.to_csv(output_file_path, header=False, index=False)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss