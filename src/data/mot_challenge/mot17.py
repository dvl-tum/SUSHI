import os.path as osp
import pandas as pd
import configparser


# Detection and ground truth file formats for MOT17
DET_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf')
GT_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'vis')


def get_mot_det_df_from_det(seq_name, data_root_path, config):
    """
    Load MOT detection file using detection files
    """
    det_type = config.det_file  # Detection type: e.g tracktor_prepr_det / gt / DPM / SDP / FRCNN
    seq_path = osp.join(data_root_path, seq_name)  # Sequence path
    det_file_path = osp.join(seq_path, f"det/{det_type}.txt")  # Detection file

    # Number and order of columns is always assumed to be the same
    det_df = pd.read_csv(det_file_path, header=None)
    det_df = det_df[det_df.columns[:len(DET_COL_NAMES)]]
    det_df.columns = DET_COL_NAMES

    # Bytetrack conf threshold
    if config.det_file == 'byte065':
        det_df = det_df[det_df['conf'].ge(0.65)].copy()

    # Coordinates are 1 based
    det_df['bb_left'] -= 1
    det_df['bb_top'] -= 1

    # If id already contains an ID assignment (e.g. using tracktor output), keep it
    if len(det_df['id'].unique()) > 1:
        det_df['preprocessor_id'] = det_df['id']
    det_df['id'] = -1  # Erase the id column

    # Include frame paths into the dataframe
    det_df['frame_path'] = det_df['frame'].apply(lambda frame_num: osp.join(seq_path, f'img1/{int(frame_num):06}.jpg'))

    assert osp.exists(det_df['frame_path'].iloc[0])  # Sanity check

    # Build scene info dictionary
    info_file_path = osp.join(seq_path, 'seqinfo.ini')
    cp = configparser.ConfigParser()
    cp.read(info_file_path)
    seq_info_dict = {'seq': seq_name,
                     'seq_path': seq_path,
                     'det_file_name': det_type,
                     'frame_height': int(cp.get('Sequence', 'imHeight')),
                     'frame_width': int(cp.get('Sequence', 'imWidth')),
                     'seq_len': int(cp.get('Sequence', 'seqLength')),
                     'fps': int(cp.get('Sequence', 'frameRate')),
                     'has_gt': osp.exists(osp.join(seq_path, 'gt')),
                     'is_gt': False}

    return det_df, seq_info_dict


def get_mot_det_df_from_gt(seq_name, data_root_path, config):
    """
    Load MOT detection file using ground truth as detections
    """
    seq_path = osp.join(data_root_path, seq_name)  # Sequence path
    det_file_path = osp.join(seq_path, "gt/gt.txt")   # Ground truth file

    # Read the gt file and assign the column names
    det_df = pd.read_csv(det_file_path, header=None)
    det_df = det_df[det_df.columns[:len(GT_COL_NAMES)]]
    det_df.columns = GT_COL_NAMES

    # Coordinates are 1 based
    det_df['bb_left'] -= 1
    det_df['bb_top'] -= 1

    # Clean out the "other classes" and use only considered gt detections
    det_df = det_df[det_df['label'].isin([1])].copy()  # Only pedestrian class  ## For now, pedestrian on vehicles are ignored.
    det_df = det_df[det_df['conf'].isin([1])].copy()  # Only considered instances. In other words, remove ignored

    # Drop the gt boxes with low visibility
    det_df = det_df[det_df['vis'] >= config.gt_training_min_vis]

    # Include frame paths into the dataframe
    det_df['frame_path'] = det_df['frame'].apply(lambda frame_num: osp.join(seq_path, f'img1/{frame_num:06}.jpg'))

    # Build scene info dictionary
    info_file_path = osp.join(seq_path, 'seqinfo.ini')
    cp = configparser.ConfigParser()
    cp.read(info_file_path)
    seq_info_dict = {'seq': seq_name,
                     'seq_path': seq_path,
                     'det_file_name': 'gt',
                     'frame_height': int(cp.get('Sequence', 'imHeight')),
                     'frame_width': int(cp.get('Sequence', 'imWidth')),
                     'seq_len': int(cp.get('Sequence', 'seqLength')),
                     'fps': int(cp.get('Sequence', 'frameRate')),
                     'has_gt': True,
                     'is_gt': True}

    return det_df, seq_info_dict


def get_mot_gt(seq_name, data_root_path, config):
    """
    Load MOT ground truth file
    """
    seq_path = osp.join(data_root_path, seq_name)  # Sequence path
    gt_file_path = osp.join(seq_path, "gt/gt.txt")  # Ground truth file

    # Read the gt file and assign the column names
    gt_df = pd.read_csv(gt_file_path, header=None)
    gt_df = gt_df[gt_df.columns[:len(GT_COL_NAMES)]]
    gt_df.columns = GT_COL_NAMES

    # Coordinates are 1 based
    gt_df['bb_left'] -= 1
    gt_df['bb_top'] -= 1

    # Clean out unnecessary classes
    gt_df = gt_df[gt_df['label'].isin([1, 2, 7, 8, 12])].copy()  # Classes 7, 8, 12 are 'ambiguous' and are not penalized. Let's keep them for now.

    # Extra bbox values that will be used for id matching
    gt_df['bb_bot'] = (gt_df['bb_top'] + gt_df['bb_height']).values
    gt_df['bb_right'] = (gt_df['bb_left'] + gt_df['bb_width']).values

    return gt_df
