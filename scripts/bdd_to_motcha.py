from PIL import Image
import os.path as osp
import os
import shutil
import configparser
import json
import pandas as pd
from tqdm import tqdm

copy = True
create = True

root = ''  # TODO: Your root folder
target = ''  # TODO: Your target folder
splits = ['train', 'val', 'test']
det_file = 'dets_bdd_byte.txt'



for split in splits:
    print("PROCESSING: ", split)
    # Create split directory at target
    if create:
        os.makedirs(osp.join(target, split))

    # Roots
    split_seqs_labels_root = osp.join(root, 'labels', split)
    split_seqs_dets_root = osp.join(root, 'dets', split)

    
    split_seqs_root = osp.join(root, 'images', split)
    split_seqs_target = osp.join(target, split)
    seqs = sorted(os.listdir(split_seqs_root))
    for seq in tqdm(seqs):
        seq_target = osp.join(split_seqs_target, seq)
        if create:
            os.makedirs(seq_target)
            os.makedirs(osp.join(seq_target, 'det'))
            os.makedirs(osp.join(seq_target, 'img1'))
            if split in ('train', 'val'):
                os.makedirs(osp.join(seq_target, 'gt'))

        # IMAGES - Move and rename
        seq_image_root = osp.join(split_seqs_root, seq)
        frames = sorted(os.listdir(seq_image_root))
        for frame in frames:
            frame_no = frame.split('-')[-1]
            frame_path = osp.join(seq_image_root, frame)
            frame_target = osp.join(seq_target, 'img1', frame_no)
            if copy:
                shutil.copy(frame_path, frame_target)

        # SEQINFO
        seqinfo_target = osp.join(seq_target, 'seqinfo.ini')
        cp = configparser.ConfigParser()
        cp.add_section('Sequence')
        cp.set('Sequence', 'name', seq)
        cp.set('Sequence', 'imDir', 'img1')
        cp.set('Sequence', 'frameRate', '5')
        cp.set('Sequence', 'seqLength', str(len(frames)))
        im = Image.open(frame_path)
        imWidth, imHeight = im.size
        cp.set('Sequence', 'imWidth', str(int(imWidth)))
        cp.set('Sequence', 'imHeight', str(int(imHeight)))
        cp.set('Sequence', 'imExt', '.jpg')
        if create:
            with open(seqinfo_target, 'w') as configfile:
                cp.write(configfile)
        
        # DETS
        seq_det = osp.join(split_seqs_dets_root, seq, det_file)
        seq_det_target = osp.join(seq_target, 'det', det_file)
        if copy:
            shutil.copy(seq_det, seq_det_target)

        # LABELS
        if split in ('train', 'val'):
            seq_label = osp.join(split_seqs_labels_root, seq+'.json')
            seq_label_target = osp.join(seq_target, 'gt', 'gt.txt')
            gt_df = pd.DataFrame(columns=['frame', 'id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'conf', 'cls', 'vis'])
            # Process the labels

            BDD_NAME_MAPPING = {
                "pedestrian": "1",
                "rider": "2",
                "car": "3",
                "truck": "4", 
                "bus": "5",
                "train": "6",
                "motorcycle": "7",
                "bicycle": "8"
            }

            with open(seq_label) as f:
                label_json = json.load(f)
                for f in label_json:
                    frame_number = int(f['name'].split('-')[-1].replace('.jpg', ''))
                    for o in f['labels']:
                        if o['category'] not in BDD_NAME_MAPPING.keys():
                            continue
                        o_id = o['id']
                        o_bbox_x = int(o['box2d']['x1'])
                        o_bbox_y = int(o['box2d']['y1'])
                        o_bbox_w = int(o['box2d']['x2'] - o['box2d']['x1'])
                        o_bbox_h = int(o['box2d']['y2'] - o['box2d']['y1'])
                        o_conf = int(not o['attributes']['crowd'])
                        o_cls = BDD_NAME_MAPPING[o['category']]
                        o_vis = 1
                        o_row = {
                            'frame': frame_number,
                            'id': o_id,
                            'bbox_x': o_bbox_x,
                            'bbox_y': o_bbox_y,
                            'bbox_w': o_bbox_w,
                            'bbox_h': o_bbox_h,
                            'conf': o_conf,
                            'cls': o_cls,
                            'vis': o_vis,
                        }
                        gt_df = pd.concat([gt_df, pd.DataFrame([o_row])], axis=0, ignore_index=True)
                
                if create:
                    gt_df.to_csv(seq_label_target, header=False, index=False)

    # SEQMAPS
    seqmap_name = 'bdd-' + split + '-all' + '.txt'
    lines = ['name'] + seqs
    seqmap_file = osp.join(target, 'seqmaps', seqmap_name)
    if create:
        os.makedirs(target, 'seqmaps', exist_ok=True)
        if osp.exists(seqmap_file):
            os.remove(seqmap_file)
        with open(seqmap_file, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
