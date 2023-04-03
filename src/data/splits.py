import os.path as osp


def get_seqs_from_splits(data_path, train_split=None, val_split=None, test_split=None):
    """
    Get splits that will be used in the experiment
    """
    _SPLITS = {}

    # MOT17 Dets
    mot17_dets = ('SDP', 'FRCNN', 'DPM')


    # Full MOT17-Train
    _SPLITS['mot17-train-all'] = {'MOT17/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 4, 5, 9, 10, 11, 13) for det in mot17_dets]}
    _SPLITS['mot17-train-split1'] = {'MOT17/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (4, 5, 9, 11) for det in mot17_dets]}
    _SPLITS['mot17-val-split1'] = {'MOT17/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 10, 13) for det in mot17_dets]}
    _SPLITS['mot17-train-split2'] = {'MOT17/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 5, 9, 10, 13) for det in mot17_dets]}
    _SPLITS['mot17-val-split2'] = {'MOT17/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (4, 11) for det in mot17_dets]}
    _SPLITS['mot17-train-split3'] = {'MOT17/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 4, 10, 11, 13) for det in mot17_dets]}
    _SPLITS['mot17-val-split3'] = {'MOT17/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (5, 9) for det in mot17_dets]}

    # MOT 17 test set
    _SPLITS['mot17-test-all'] = {
        'MOT17/test': [f'MOT17-{seq_num:02}-{det}' for seq_num in (1, 3, 6, 7, 8, 12, 14) for det in mot17_dets]}

    #######
    # MOT20
    #######
    _SPLITS['mot20-train-all'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (1, 2, 3, 5)]}
    _SPLITS['mot20-test-all'] = {'MOT20/test': [f'MOT20-{seq_num:02}' for seq_num in (4, 6, 7, 8)]}

    _SPLITS['mot20-train-split1'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (1, 2, 3)]}
    _SPLITS['mot20-train-split2'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (1, 2, 5)]}
    _SPLITS['mot20-train-split3'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (1, 3, 5)]}
    _SPLITS['mot20-train-split4'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (2, 3, 5)]}

    _SPLITS['mot20-val-split1'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (5,)]}
    _SPLITS['mot20-val-split2'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (3,)]}
    _SPLITS['mot20-val-split3'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (2,)]}
    _SPLITS['mot20-val-split4'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (1,)]}



    #######
    # DanceTrack
    #######
    dancetrack_train_seqs = (1, 2, 6, 8, 12, 15, 16, 20, 23, 24, 27, 29, 32, 33, 37, 39, 44, 45, 49, 51, 52, 53, 55, 57, 61, 62, 66, 68, 69, 72, 74, 75, 80, 82, 83, 86, 87, 96, 98, 99)
    dancetrack_val_seqs = (4, 5, 7, 10, 14, 18, 19, 25, 26, 30, 34, 35, 41, 43, 47, 58, 63, 65, 73, 77, 79, 81, 90, 94, 97)
    dancetrack_test_seqs = (3, 9, 11, 13, 17, 21, 22, 28, 31, 36, 38, 40, 42, 46, 48, 50, 54, 56, 59, 60, 64, 67, 70, 71, 76, 78, 84, 85, 88, 89, 91, 92, 93, 95, 100)

    assert set(dancetrack_train_seqs + dancetrack_val_seqs + dancetrack_test_seqs) == set([x for x in range(1, 101)]), "Missing sequence in the dancetrack splits"
    assert len(dancetrack_train_seqs + dancetrack_val_seqs + dancetrack_test_seqs) == 100, "Missing or duplicate sequence in the dancetrack splits"

    _SPLITS['dancetrack-train-all'] = {'DANCETRACK/train': [f'dancetrack{seq_num:04}' for seq_num in dancetrack_train_seqs]}
    _SPLITS['dancetrack-val-all'] = {'DANCETRACK/val': [f'dancetrack{seq_num:04}' for seq_num in dancetrack_val_seqs]}
    _SPLITS['dancetrack-test-all'] = {'DANCETRACK/test': [f'dancetrack{seq_num:04}' for seq_num in dancetrack_test_seqs]}

    _SPLITS['dancetrack-debug'] = {'DANCETRACK/train': [f'dancetrack{seq_num:04}' for seq_num in (1,)]}
    _SPLITS['dancetrack-val-debug'] = {'DANCETRACK/val': [f'dancetrack{seq_num:04}' for seq_num in (4,)]}


    ########
    # BDD
    ########
    _SPLITS['bdd-val-debug'] = {'BDD/val': [f'{seq_name}' for seq_name in ('b1c66a42-6f7d68ca', 'b1c9c847-3bda4659')]}


    # Ensure that split is valid
    assert train_split in _SPLITS.keys() or train_split is None, "Training split is not valid!"
    assert val_split in _SPLITS.keys() or val_split is None, "Validation split is not valid!"
    assert test_split in _SPLITS.keys() or test_split is None, "Test split is not valid!"

    # Get the sequences to use in the experiment
    seqs = {}
    if train_split is not None:
        seqs['train'] = {osp.join(data_path, seqs_path): seq_list for seqs_path, seq_list in
                         _SPLITS[train_split].items()}
    if val_split is not None:
        seqs['val'] = {osp.join(data_path, seqs_path): seq_list for seqs_path, seq_list in
                       _SPLITS[val_split].items()}
    if test_split is not None:
        seqs['test'] = {osp.join(data_path, seqs_path): seq_list for seqs_path, seq_list in
                        _SPLITS[test_split].items()}
    return seqs, (train_split, val_split, test_split)
