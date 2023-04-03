import argparse
import torch
import os
import os.path as osp
from datetime import datetime
import math

def _store_bool(num):
    return bool(int(num))

def _isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k,v = kv.split("=")
            if v.isdigit():
                my_dict[k] = int(v)
            
            elif _isfloat(v):
                my_dict[k] = float(v)
            
            elif v == 'False':
                my_dict[k] = False

            elif v == 'True':
                my_dict[k] = True

            else:
                my_dict[k] = v

        setattr(namespace, self.dest, my_dict)


def get_arguments(args=None):
    parser = argparse.ArgumentParser()

    # EXPERIMENT
    parser.add_argument('--experiment_mode', help='train/test/train-cval/eval', type=str, required=True)  # What kind of an experiment will be conducted
    parser.add_argument('--run_id', help='identifier string for current experiment', type=str, default=None)
    parser.add_argument('--connectivity', help='chunk or full', type=str, default='chunk')

    # PATHS
    parser.add_argument('--data_path', help='Where is data located?', type=str,
                        default='/storage/slurm/cetintas')
    parser.add_argument('--output_path', help='Where is the output folder?', type=str,
                        default='/storage/user/cetintas/SUSHI/output')

    parser.set_defaults(mot_sub_folder='mot_files')  # Tracker output for sequences will be stored here

    # SEED
    parser.add_argument('--seed', help='Seed of the experiment', type=int, default=0)

    # DEVICE SPECS
    parser.add_argument('--cuda', help='Run on gpu', dest='cuda', action='store_true')
    parser.add_argument('--num_workers', help='Number of workers', type=int, default=4)

    # SPLITS
    parser.add_argument('--train_splits', type=str, nargs='*', help='Training split', default=[None])
    parser.add_argument('--val_splits', type=str, nargs='*', help='Validation split', default=[None])
    parser.add_argument('--test_splits', type=str, nargs='*', help='Test split', default=[None])
    parser.add_argument('--cval_seqs', help='All range that crossvalidation covers (e.g all train seqs)', type=str, default='mot17-train-all')

    # DETECTIONS
    parser.add_argument('--det_file', help='Detection file to use', type=str, default='tracktor_prepr_det')
    parser.set_defaults(gt_training_min_vis=0.0)  # Minimum visibility score allowed in a GT box to be used for training
    parser.set_defaults(gt_assign_min_iou=0.5)  # Min IoU between a GT and detected box so that an assignment is allowed

    # PRETRAINED MODELS
    parser.add_argument('--save_cp', help='Save checkpoints', dest='save_cp', action='store_true')
    parser.add_argument('--feature_embedding_model_path', help='Reid model path to obtain features', type=str,
                        default='')
    parser.add_argument('--hicl_model_path', help='HICL model path to perform tracking', type=str,
                        default='')
    
    parser.add_argument('--load_train_ckpt', action='store_true', default=False)
    
    
    # WEIGHT SHARING
    parser.add_argument('--share_weights', help='weight sharing scheme used', type=str, default='all_but_first')
    parser.add_argument('--node_level_embed', action='store_true', default=False)  
    parser.add_argument('--edge_level_embed', action='store_true', default=False)  


    # REID SETTING    
    parser.add_argument('--reid_arch', help='Reid model path to obtain features', type=str,
                        default='fastreid_msmt_BOT_R50_ibn')
    parser.add_argument('--reid_sim_fn', help='Reid model path to obtain features', type=str,
                        default='l2')
    parser.add_argument('--edge_sim_fn', help='Reid model path to obtain features', type=str,
                        default='l2')

    parser.add_argument('--node_dim', help='Reid model path to obtain features', type=int,
                        default=2048)

    parser.add_argument('--l2_norm_reid',action='store_true', default=False)
    

    # EMBEDDING DIRECTORIES
    parser.add_argument('--reid_embeddings_dir', help='Storage directory of reid embeddings', type=str, default='reid')
    parser.add_argument('--node_embeddings_dir', help='Storage directory of node embeddings', type=str, default='node')

    # TRAINING
    parser.add_argument('--num_epoch', help='Number of epochs for training', type=int, default=250)
    parser.add_argument('--num_batch', help='Number of graphs per batch', type=int, default=8)
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.0003)  # MPNTrack param
    parser.add_argument('--gamma', help='Focal loss gamma parameter', type=float, default=1)
    parser.add_argument('--weight_decay', help='Weight decay', type=float, default=0.0001)  # MPNTrack param
    parser.add_argument('--augmentation', help='Perform data augmentation during training', dest='augmentation', action='store_true')
    parser.add_argument('--train_dataset_frame_overlap', help='Most frames to overlap for different datapoints', type=int, default=20)
    parser.add_argument('--start_eval', help='Epoch at which evaluation will start', type=int, default=100) # Don't start evaluating bf all layers are unfrozen!
    parser.add_argument('--no_fp_loss', help='Do not compute a loss for FPs', dest='no_fp_loss', action='store_true', default=False) 


    # augmentation
    parser.set_defaults(min_iou_bb_wiggling=0.85)  # Minimum IoU w.r.t. original box used when doing
    parser.set_defaults(min_ids_to_drop_perc=0)  # Minimum percentage of ids s.t. all of its detections will be dropped
    parser.set_defaults(max_ids_to_drop_perc=0.1)  # Maximum percentage of ids s.t. all of its detections will be dropped
    parser.set_defaults(min_detects_to_drop_perc=0)  # Minimum Percentage of detections that might be randomly dropped
    parser.set_defaults(max_detects_to_drop_perc=0.2)  # Maximum Percentage of detections that might be randomly dropped

    # EVALUATION
    parser.set_defaults(evaluation_graph_overlap_ratio=0.5)  # During evaluation this ratio of frames will overlap
    parser.add_argument('--min_track_len', help='Shorter tracks will be removed', type=int, default=2)

    parser.add_argument('--rounding_method', help='Flow formulation solver', type=str, default='exact')  # Determines whether an LP is used for rounding ('exact') or a greedy heuristic ('greedy')
    parser.set_defaults(solver_backend='pulp')  # Determines package used to solve the LP, (Gurobi requires a license, pulp does not)
    parser.add_argument('--no_interpolate', help='Do not interpolate missing detections of trajectories', dest='no_interpolate', action='store_true')


    # GRAPH PARAMETERS
    parser.add_argument('--top_k_nns', help='Similarity of nodes to connect in a graph', type=int, default=10)
    parser.add_argument('--frames_per_graph', help='Total number of frames to process', type=int, default=512)  # If larger than seq length, all frames of the seq will be used.
    parser.add_argument('--frames_per_level', type=int, nargs='*', help='Number of frames to process at each hierarchical level', default=[2, 4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument('--pruning_method', type=str, nargs='*', help='Pruning scheme used at each hierarchical level', default=['geometry', 'motion_005', 'motion_005', 'motion_005', 'motion_005', 'motion_005', 'motion_005', 'motion_005', 'motion_005'])

    parser.set_defaults(symmetric_edges=True)  # Graphs contain each edge twice with changing source and destination

    # HIERARCHICAL PARAMETERS
    parser.add_argument('--hicl_depth', help='The depth of the hierarchical architecture', type=int, default=9)
    parser.add_argument('--node_id_min_ratio', help='Min percentage of the most common id for hicl layers gt assignment', type=float, default=0.5)
    parser.add_argument('--depth_pretrain_iteration', help='Number of iterations before unlocking next level', type=int,
                        default=500)

    # MOTION SETTING
    parser.add_argument('--motion_max_length', type=int, nargs='*', help='Maximum number of frames used to encode trajectories before pred at each layer', default=[2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument('--motion_pred_length', type=int, nargs='*', help='Number of future/past frames for which locations are predicted at each level', default=[2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument('--interpolate_motion', action='store_true', help='If true, missing locations in each node location are filled via linear interpolation', default=False)
    parser.add_argument('--linear_center_only', action='store_true',help='If true, the linear motion model will only predict center locations, and leave width/height constant', default=False)

    # Hierarchical Features
    parser.add_argument('--do_hicl_feats', action='store_true', default=False)
    parser.add_argument("--hicl_feats_args", dest="hicl_feats_args", action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL", default={})

    
    # ReID 
    parser.add_argument('--reid_img_h', help='Height to which ReID box images are resized', type=int, default=256)
    parser.add_argument('--reid_img_w', help='Height to which ReID box images are resized', type=int, default=128)
    parser.add_argument('--zero_nodes', action='store_true', default=False)

    # MOTION FEATURES
    parser.add_argument('--mpn_use_motion', type=_store_bool, nargs='*', help='Use motion predictions GIoU as edge feature at each layer', default=[False, True, True, True, True, True, True, True, True])
    parser.add_argument('--mpn_use_reid_edge', type=_store_bool, nargs='*', default=[True]*9)
    parser.add_argument('--mpn_use_pos_edge', type=_store_bool, nargs='*', default=[True]*9)

    # VERBOSE
    parser.set_defaults(verbose_iteration=10)

    # EXPERIMENT EVAL
    parser.add_argument('--old_experiments', type=str, nargs='*', help='Experiment files to merge', default=[None])


    # Postprocess
    if args is not None:
        config = post_config(parser.parse_args(args))
    else:
        config = post_config(parser.parse_args())
    

    return config


def post_config(config):
    ensure_config_consistency(config)
    config.device = torch.device("cuda:0" if config.cuda else "cpu")  # Set the device
    config = create_experiment_path(config)

    # Dataframe Columns to keep in order to produce a tracking output
    config.VIDEO_COLUMNS = ['frame_path', 'frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'bb_right', 'bb_bot']  # Columns to save in the output df
    config.TRACKING_OUT_COLS = ['frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']  # MotCha output format

    # Motion things
    config.do_motion = max(config.mpn_use_motion)

    return config


def ensure_config_consistency(config):
    """
    Make sure that user input is consistent
    """
    assert config.experiment_mode in ("train", "train-cval", "test", "eval"), "Invalid experiment mode"
    if config.experiment_mode != 'eval':
        assert config.connectivity in ("chunk", "full")
        if config.experiment_mode == "train":
            assert config.train_splits != [None] and len(config.train_splits) == 1 and len(config.val_splits) <= 1 and config.test_splits == [None], "Splits are not compatible with train mode"

        elif config.experiment_mode == "test":
            assert len(config.test_splits) == 1 and config.val_splits == [None] and config.train_splits == [None], "Splits are not compatible with test mode"

        elif config.experiment_mode == "train-cval":
            assert config.train_splits != [None] and config.val_splits != [None] and len(config.train_splits) == len(config.val_splits) and config.test_splits == [None], "Splits are nor compatible with train-cval mode"

        assert config.frames_per_graph == config.frames_per_level[-1], "Total number of frames should be equal to the frames in last level"
        assert config.hicl_depth == len(config.frames_per_level), "Depth is not equal to number of frames per level"
        for attr in ('mpn_use_motion',):
            attr_ = getattr(config, attr)
            if max(attr_):
                assert config.hicl_depth == len(attr_), f"Depth is not equal to length of {attr}"
        
        for attr in ('motion_max_length', 'motion_pred_length'):
            assert (config.hicl_depth - 1)== len(getattr(config, attr)), f"Depth -1 is not equal to length of {attr}"
    
def create_experiment_path(config):
    """
    Get unique experiment id
    """
    date = '{date:%m-%d_%H:%M:%S.%f}'.format(date=datetime.now())
    run_str = config.run_id + '_' + date if config.run_id is not None else date
    
    config.experiment_path = osp.join(config.output_path, 'experiments', run_str)
    if config.run_id is not None and config.experiment_mode != 'eval': 
        assert not osp.exists(config.experiment_path), f"Experiment id {config.experiment_path} exists! Choose another one!"

    # Create the paths
    os.makedirs(config.experiment_path)

    return config

