"""
Process the sequences from the detection/gt file. Moreover, if they are already processed, loads the corresponding
dataframes.
"""

from typing import OrderedDict
import numpy as np
import os
import os.path as osp
import shutil
import pandas as pd
from torch.utils.data import DataLoader
import torch
from lapsolver import solve_dense

from src.data.mot_challenge.mot17 import get_mot_gt, get_mot_det_df_from_gt, get_mot_det_df_from_det
from src.data.dancetrack.dancetrack import get_dancetrack_gt, get_dancetrack_det_df_from_det
from src.data.bdd.bdd import get_bdd_gt, get_bdd_det_df_from_det
from src.utils.deterministic import seed_generator, seed_worker
from src.models.reid.resnet import resnet50_fc256, resnet50_fc512, load_pretrained_weights
from src.models.reid.fastreid_models import load_fastreid_model
from src.data.misc_datasets import BoundingBoxDataset
from src.utils.graph_utils import iou
import time

# ################################### SETUP ###################################
# Loader functions for the sequence type
_SEQ_TYPE_DETS_DF_LOADER = {'MOT': get_mot_det_df_from_det, 'MOT_GT': get_mot_det_df_from_gt, 'MOT_DANCETRACK': get_dancetrack_det_df_from_det, 'MOT_BDD': get_bdd_det_df_from_det}
_SEQ_TYPE_GT_DF_LOADER = {'MOT': get_mot_gt, 'MOT_GT': get_mot_gt, 'MOT_DANCETRACK': get_dancetrack_gt, 'MOT_BDD': get_bdd_gt}

# Are boxes allowed to be outside?
_ENSURE_BOX_IN_FRAME = {'MOT': False, 'MOT_GT': False, 'MOT_DANCETRACK': False, 'MOT_BDD': False}  # MOT17 boxes are inside the frame for both det and gt

# Available sequences
_SEQ_TYPES = {}
# --------- MOT17 ---------
mot17_seqs = [f'MOT17-{seq_num:02}-{det}{extra_str}' for seq_num in (2, 4, 5, 9, 10, 11, 13) for det in ('DPM', 'SDP', 'FRCNN', 'GT') for extra_str in ('', '-train-half', '-val-half')]
mot17_seqs += [f'MOT17-{seq_num:02}-{det}' for seq_num in (1, 3, 6, 7, 8, 12, 14) for det in ('DPM', 'SDP', 'FRCNN')]
mot17_seqs += ['MOT17-00-GT']  # Debug "sequence" = first 50 frames of MOT17-02
for seq_name in mot17_seqs:
    if 'GT' in seq_name:
        _SEQ_TYPES[seq_name] = 'MOT_GT'

    else:
        _SEQ_TYPES[seq_name] = 'MOT'

#--------- MOT20 ---------
mot20_seqs = [f'MOT20-{seq_num:02}' for seq_num in (1, 2, 3, 5, 4, 6, 7, 8)]
for seq_name in mot20_seqs:
    if 'GT' in seq_name:
        _SEQ_TYPES[seq_name] = 'MOT_GT'

    else:
        _SEQ_TYPES[seq_name] = 'MOT'


#--------- DanceTrack ---------
dancetrack_seqs = [f'dancetrack{seq_num:04}' for seq_num in range(1,101)]
for seq_name in dancetrack_seqs:
    _SEQ_TYPES[seq_name] = 'MOT_DANCETRACK'


#--------- BDD ---------
bdd_seqs = [f'{seq_name}' for seq_name in ('b1c66a42-6f7d68ca', 'b1c81faa-3df17267', 'b1c81faa-c80764c5', 'b1c9c847-3bda4659')]
for seq_name in bdd_seqs:
    _SEQ_TYPES[seq_name] = 'MOT_BDD'

# -------------------------
# ###################################  END  ###################################

class DataFrameWSeqInfo(pd.DataFrame):
    """
    Class used to store each sequences's processed detections as a DataFrame. We just add a metadata atribute to
    pandas DataFrames it so that sequence metainfo such as fps, etc. can be stored in the attribute 'seq_info_dict'.
    This attribute survives serialization.
    This solution was adopted from:
    https://pandas.pydata.org/pandas-docs/stable/development/extending.html#define-original-properties
    """
    _metadata = ['seq_info_dict']

    @property
    def _constructor(self):
        return DataFrameWSeqInfo

class MOTSeqProcessor:
    """
    Class to process detections files coming from different mot_seqs.
    Main method is process_detections. It does the following:
    - Loads a DataFrameWSeqInfo (~pd.DataFrame) from a  detections file (self.det_df) via a the 'det_df_loader' func
    corresponding to the sequence type (mapped via _SEQ_TYPES)
    - Adds Sequence Info to the df (fps, img size, moving/static camera, etc.) as an additional attribute (_get_det_df)
    - If GT is available, assigns GT identities to the detected boxes via bipartite matching (_assign_gt)
    - Stores the df on disk (_store_det_df)
    - If required, precomputes CNN embeddings for every detected box and stores them on disk (_store_embeddings)

    The stored information assumes that each MOT sequence has its own directory. Inside it all processed data is
    stored as follows:
        +-- <Sequence name>
        |   +-- processed_data
        |       +-- det
        |           +-- <dataset_params['det_file_name']>.pkl # pd.DataFrame with processed detections and metainfo
        |       +-- embeddings
        |           +-- <dataset_params['det_file_name']> # Precomputed embeddings for a set of detections
        |               +-- <CNN Name >
        |                   +-- {frame1}.jpg
        |                   ...
        |                   +-- {frameN}.jpg
    """
    def __init__(self, dataset_path, seq_name, config):
        self.seq_name = seq_name
        self.dataset_path = dataset_path
        self.seq_type = _SEQ_TYPES[seq_name]
        self.det_df_loader = _SEQ_TYPE_DETS_DF_LOADER[self.seq_type]
        self.gt_df_loader = _SEQ_TYPE_GT_DF_LOADER[self.seq_type]
        self.config = config

    def _ensure_boxes_in_frame(self):
        """
        Determines whether boxes are allowed to have some area outside the image (all GT annotations in MOT15 are inside
        the frame hence we crop its detections to also be inside it)
        """

        initial_bb_top = self.det_df['bb_top'].values.copy()
        initial_bb_left = self.det_df['bb_left'].values.copy()

        self.det_df['bb_top'] = np.maximum(self.det_df['bb_top'].values, 0).astype(int)
        self.det_df['bb_left'] = np.maximum(self.det_df['bb_left'].values, 0).astype(int)

        bb_top_diff = self.det_df['bb_top'].values - initial_bb_top
        bb_left_diff = self.det_df['bb_left'].values - initial_bb_left

        self.det_df['bb_height'] -= bb_top_diff
        self.det_df['bb_width'] -= bb_left_diff

        img_height, img_width = self.det_df.seq_info_dict['frame_height'], self.det_df.seq_info_dict['frame_width']
        self.det_df['bb_height'] = np.minimum(img_height - self.det_df['bb_top'], self.det_df['bb_height']).astype(int)
        self.det_df['bb_width'] = np.minimum(img_width - self.det_df['bb_left'], self.det_df['bb_width']).astype(int)

    def _sanity_check_boxes(self):
        # Sanity check that boxes do not lay completely outside
        frame_height, frame_width = self.det_df.seq_info_dict['frame_height'], self.det_df.seq_info_dict['frame_width']
        conds = (self.det_df['bb_width'] > 0) & (self.det_df['bb_height'] > 0)
        conds = conds & (self.det_df['bb_right'] > 0) & (self.det_df['bb_bot'] > 0)
        conds = conds & (self.det_df['bb_left'] < frame_width) & (self.det_df['bb_top'] < frame_height)
        assert self.det_df.equals(self.det_df[conds].copy()), "There are bounding boxes outside of the frame!"

    def _add_extra_det_features(self):
        """
        Create additional features for each detection. (e.g bbox centre, area etc.)
        """
        self.det_df['bb_bot'] = (self.det_df['bb_top'] + self.det_df['bb_height']).values
        self.det_df['bb_right'] = (self.det_df['bb_left'] + self.det_df['bb_width']).values
        self.det_df['feet_x'] = self.det_df['bb_left'] + 0.5 * self.det_df['bb_width']
        self.det_df['feet_y'] = self.det_df['bb_top'] + self.det_df['bb_height']

    def _get_dfs(self):
        """
        Load a pd.Dataframe with each entry corresponding to a detection. Same for the ground truth file.
        """
        # Read the dfs
        self.det_df, seq_info_dict = self.det_df_loader(self.seq_name, self.dataset_path, self.config)
        if seq_info_dict['has_gt']:
            self.gt_df = self.gt_df_loader(self.seq_name, self.dataset_path, self.config)

        # Copy the dataframe into our class
        self.det_df = DataFrameWSeqInfo(self.det_df)
        self.det_df.seq_info_dict = seq_info_dict

        # Ensure the bboxes are in the frame
        if self.seq_type in _ENSURE_BOX_IN_FRAME and _ENSURE_BOX_IN_FRAME[self.seq_type]:
            self._ensure_boxes_in_frame()

        if self.config.det_file in ('tracktor_prepr_det', 'aplift'):
            if hasattr(self, 'gt_df'):
                initial_bb_top = self.gt_df['bb_top'].values.copy()
                initial_bb_left = self.gt_df['bb_left'].values.copy()
                
                self.gt_df['bb_top'] = np.maximum(self.gt_df['bb_top'].values, 0).astype(int)
                self.gt_df['bb_left'] = np.maximum(self.gt_df['bb_left'].values, 0).astype(int)
                
                bb_top_diff = self.gt_df['bb_top'].values - initial_bb_top
                bb_left_diff = self.gt_df['bb_left'].values - initial_bb_left
                
                self.gt_df['bb_height'] -= bb_top_diff
                self.gt_df['bb_width'] -= bb_left_diff
                
                img_height, img_width = seq_info_dict['frame_height'], seq_info_dict['frame_width']
                self.gt_df['bb_height'] = np.minimum(img_height - self.gt_df['bb_top'], self.gt_df['bb_height']).astype(int)
                self.gt_df['bb_width'] = np.minimum(img_width - self.gt_df['bb_left'], self.gt_df['bb_width']).astype(int)


        # Add extra measurements
        self._add_extra_det_features()

        # Sanity check that bboxes are within the frame
        self._sanity_check_boxes()

        # Sort the detections and assign unique detection ids
        self.det_df.sort_values(by='frame', inplace=True)
        self.det_df['detection_id'] = np.arange(self.det_df.shape[0])  # Unique detection ids

    def _assign_gt(self):
        """
        Assigns a GT identity to every detection in self.det_df, based on the ground truth boxes in self.gt_df.
        The assignment is done frame by frame via bipartite matching.
        """
        if self.det_df.seq_info_dict['has_gt'] and not self.det_df.seq_info_dict['is_gt']:
            print(f"Assigning ground truth identities to detections to sequence {self.seq_name}")
            for frame in self.det_df['frame'].unique():
                frame_detects = self.det_df[self.det_df.frame == frame]
                frame_gt = self.gt_df[self.gt_df.frame == frame]

                # Compute IoU for each pair of detected / GT bounding box
                iou_matrix = iou(frame_detects[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values,
                                 frame_gt[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values)

                iou_matrix[iou_matrix < self.config.gt_assign_min_iou] = np.nan
                dist_matrix = 1 - iou_matrix
                assigned_detect_ixs, assigned_detect_ixs_ped_ids = solve_dense(dist_matrix)
                unassigned_detect_ixs = np.array(list(set(range(frame_detects.shape[0])) - set(assigned_detect_ixs)))

                assigned_detect_ixs_index = frame_detects.iloc[assigned_detect_ixs].index
                assigned_detect_ixs_ped_ids = frame_gt.iloc[assigned_detect_ixs_ped_ids]['id'].values
                unassigned_detect_ixs_index = frame_detects.iloc[unassigned_detect_ixs].index

                self.det_df.loc[assigned_detect_ixs_index, 'id'] = assigned_detect_ixs_ped_ids
                self.det_df.loc[unassigned_detect_ixs_index, 'id'] = -1  # False Positives

    def _store_dfs(self):
        """
        Save detection and ground truth dataframes under processed data
        """
        # Storage dirs
        processed_dets_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data', 'det')
        # Create dirs
        os.makedirs(processed_dets_path, exist_ok=True)
        # File names
        det_df_path = osp.join(processed_dets_path, self.config.det_file + '.pkl')
        # Store
        self.det_df.to_pickle(det_df_path)

        # Repeat for gt
        if self.det_df.seq_info_dict['has_gt']:
            processed_gt_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data', 'gt')
            os.makedirs(processed_gt_path, exist_ok=True)
            gt_df_path = osp.join(processed_gt_path, 'gt_df' + '.pkl')
            self.gt_df.to_pickle(gt_df_path)

    def _store_embeddings(self):
        """
        Stores node and reid embeddings corresponding for each detection in the given sequence.
        Embeddings are stored at:
        Essentially, each set of processed detections (e.g. raw, prepr w. frcnn, prepr w. tracktor) has a storage path, corresponding
        to a detection file (det_file_name). Within this path, different CNNs, have different directories
        (specified in dataset_params['node_embeddings_dir'] and dataset_params['reid_embeddings_dir']), and within each
        directory, we store pytorch tensors corresponding to the embeddings in a given frame, with shape
        (N, EMBEDDING_SIZE), where N is the number of detections in the frame.
        """
        assert self.feature_embedding_model is not None
        assert self.config.reid_embeddings_dir is not None and self.config.node_embeddings_dir

        # Directory paths
        node_embeds_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data/embeddings',
                                   self.config.det_file, self.config.node_embeddings_dir)

        reid_embeds_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data/embeddings',
                                    self.config.det_file, self.config.reid_embeddings_dir)

        # Delete if exists, and create the directories
        if osp.exists(node_embeds_path):
            print("Found existing stored node embeddings. Deleting them and replacing them for new ones")
            shutil.rmtree(node_embeds_path)
        if osp.exists(reid_embeds_path):
            print("Found existing stored reid embeddings. Deleting them and replacing them for new ones")
            shutil.rmtree(reid_embeds_path)
        os.makedirs(node_embeds_path)
        os.makedirs(reid_embeds_path)

        print(f"Computing embeddings for {self.det_df.shape[0]} detections")  # Info num detections

        # Make sure that we don't run out of memory, so batch the detections if necessary
        num_dets = self.det_df.shape[0]
        max_dets_per_df = int(1e5)
        frame_cutpoints = [self.det_df.frame.iloc[i] for i in np.arange(0, num_dets, max_dets_per_df, dtype=int)]
        frame_cutpoints += [self.det_df.frame.iloc[-1] + 1]

        
        t = 0
        # Compute and store embeddings
        for frame_start, frame_end in zip(frame_cutpoints[:-1], frame_cutpoints[1:]):
            # Get the corresponding frames
            sub_df_mask = self.det_df.frame.between(frame_start, frame_end - 1)
            sub_df = self.det_df.loc[sub_df_mask]

            # Dataloader
            bbox_dataset = BoundingBoxDataset(sub_df, seq_info_dict=self.det_df.seq_info_dict,
                                              return_det_ids_and_frame=True, 
                                              transforms=self.transforms,
                                              output_size=(self.config.reid_img_h, self.config.reid_img_w))
            bbox_loader = DataLoader(bbox_dataset, batch_size=1000, pin_memory=True,
                                     num_workers=self.config.num_workers,
                                     worker_init_fn=seed_worker, generator=seed_generator(),)

            # Feed them to the model
            self.feature_embedding_model.eval()
            node_embeds, reid_embeds = [], []  # Node: before fc layers (2048), reid after fc layers (256)
            frame_nums, det_ids = [], []
            with torch.no_grad():
                for frame_num, det_id, bboxes in bbox_loader:
                    #node_out, reid_out = self.feature_embedding_model(bboxes.to(self.config.device))
                    feature_out = self.feature_embedding_model(bboxes.to(self.config.device))
                    if isinstance(feature_out, torch.Tensor):
                        node_out = feature_out
                        reid_out = feature_out.clone()
                    else:
                        node_out, reid_out = feature_out
                        
                    node_embeds.append(node_out.cpu())
                    reid_embeds.append(reid_out.cpu())
                    frame_nums.append(frame_num)
                    det_ids.append(det_id)

            # Merge with all results
            det_ids = torch.cat(det_ids, dim=0)
            frame_nums = torch.cat(frame_nums, dim=0)
            node_embeds = torch.cat(node_embeds, dim=0)
            reid_embeds = torch.cat(reid_embeds, dim=0)

            # Add detection ids as first column of embeddings, to ensure that embeddings are loaded correctly
            node_embeds = torch.cat((det_ids.view(-1, 1).float(), node_embeds), dim=1)
            reid_embeds = torch.cat((det_ids.view(-1, 1).float(), reid_embeds), dim=1)

            # Save embeddings grouped by frame
            for frame in sub_df.frame.unique():
                mask = frame_nums == frame
                frame_node_embeds = node_embeds[mask]
                frame_reid_embeds = reid_embeds[mask]

                frame_node_embeds_path = osp.join(node_embeds_path, f"{frame}.pt")
                frame_reid_embeds_path = osp.join(reid_embeds_path, f"{frame}.pt")

                torch.save(frame_node_embeds, frame_node_embeds_path)
                torch.save(frame_reid_embeds, frame_reid_embeds_path)

            # print("Finished storing embeddings")
        print("Finished computing and storing embeddings")

    def process_detections(self):
        """
        Main processing function.
        Load the dataframe > Assign gt > Store df > Store embeddings
        """
        self._get_dfs()  # Read the detection and ground truth files
        self._assign_gt()  # Assign ground truth ids
        self._store_dfs()  # Store the detection and gt dframes
        self._store_embeddings()  # Store reid and node embeddings

        return self.det_df

    def _is_dets_and_embeds_ok(self, seq_path, seq_det_df_path):
        # Verify the processed detections file
        node_embeds_path = osp.join(seq_path, 'processed_data/embeddings', self.config.det_file,
                                    self.config.node_embeddings_dir)
        reid_embeds_path = osp.join(seq_path, 'processed_data/embeddings', self.config.det_file,
                                    self.config.reid_embeddings_dir)
        try:
            num_frames = len(pd.read_pickle(seq_det_df_path)['frame'].unique())
            processed_dets_exist = True
        except:
            num_frames = -1
            processed_dets_exist = False

        # Verify the length of the embeddings
        embeds_ok = osp.exists(node_embeds_path) and len(os.listdir(node_embeds_path)) == num_frames
        embeds_ok = embeds_ok and osp.exists(reid_embeds_path) and len(os.listdir(reid_embeds_path)) == num_frames

        # Are both okay?
        return processed_dets_exist and embeds_ok

    def _load_feature_embedding_model(self):
        """
        Load the embedding cnn model to get the embeddings
        """
        transforms = None

        print("REID ARCH??")
        if self.config.reid_arch == 'resnet50_fc512':
            print("RESNET 50 fc512!!")
            feature_embedding_model = resnet50_fc512(num_classes=1000, loss='xent', pretrained=True).to(self.config.device)
            load_pretrained_weights(feature_embedding_model, self.config.feature_embedding_model_path)

        elif self.config.reid_arch.startswith('fastreid_'):
            print("FASTREID MODEL!!")
            feature_embedding_model, transforms =  load_fastreid_model(self.config.reid_arch)

        elif self.config.reid_arch == 'old_model':
            print("OLD MODEL!!")

            #feature_embedding_model = resnet50_fc256(num_classes=2220, loss='xent', pretrained=True).to(self.config.device)
            model_cls = resnet50_fc256 if 'duke' in self.config.feature_embedding_model_path else resnet50_fc512
            num_classes = 2220 if 'duke' in self.config.feature_embedding_model_path else 2968
            feature_embedding_model = model_cls(num_classes=num_classes, loss='xent', pretrained=True).to(self.config.device)
            load_pretrained_weights(feature_embedding_model, self.config.feature_embedding_model_path)
        
        else:
            raise NameError(f"ReID architecture is not {self.config.reid_arch} a valid option")
            
        #load_pretrained_weights(feature_embedding_model, self.config.feature_embedding_model_path)
        return feature_embedding_model, transforms

    def load_or_process_detections(self):
        """
        Tries to load a set of processed detections if it's safe to do so. otherwise, it processes them and stores the
        result
        """

        # Paths
        seq_path = osp.join(self.dataset_path, self.seq_name)
        seq_det_df_path = osp.join(seq_path, 'processed_data/det', self.config.det_file + '.pkl')

        if self._is_dets_and_embeds_ok(seq_path, seq_det_df_path):
            print(f"Loading processed dets for sequence {self.seq_name} from {seq_det_df_path}")
            seq_det_df = pd.read_pickle(seq_det_df_path).reset_index().sort_values(by=['frame', 'detection_id'])

        else:
            print(f'Detections for sequence {self.seq_name} need to be processed. Starting processing')
            self.feature_embedding_model, self.transforms = self._load_feature_embedding_model()
            seq_det_df = self.process_detections()

        seq_det_df.seq_info_dict['seq_path'] = seq_path
        return seq_det_df


