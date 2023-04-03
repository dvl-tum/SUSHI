from configs.config import get_arguments
from src.utils.deterministic import make_deterministic
from src.tracker.hicl_tracker import HICLTracker
from src.data.splits import get_seqs_from_splits
import os.path as osp
from TrackEval.scripts.run_mot_challenge import evaluate_mot17
import time
import torch

if __name__ == "__main__":
    config = get_arguments()  # Get hyperparameters
    make_deterministic(config.seed)  # Make the experiment deterministic

    # Print the config and experiment id
    print("Experiment ID:", config.experiment_path)
    print("Experiment Mode:", config.experiment_mode)
    print("----- CONFIG -----")
    for key, value in vars(config).items():
        print(key, ':', value)
    print("------------------")

    # TRAINING
    if config.experiment_mode == 'train':
        # Get the splits for the experiment
        seqs, splits = get_seqs_from_splits(data_path=config.data_path, train_split=config.train_splits[0], val_split=config.val_splits[0])
        # Initialize the tracker
        hicl_tracker = HICLTracker(config=config, seqs=seqs, splits=splits)

        if config.load_train_ckpt:
            print("Loading checkpoint from ", config.hicl_model_path)
            hicl_tracker.model = hicl_tracker.load_pretrained_model()

        # Train the tracker
        hicl_tracker.train()

    # CROSS-VALIDATION
    elif config.experiment_mode == 'train-cval':
        # Each training/validation split
        for train_split, val_split in zip(config.train_splits, config.val_splits):
            # Get the splits for the experiment
            seqs, splits = get_seqs_from_splits(data_path=config.data_path, train_split=train_split, val_split=val_split)
            # Initialize the tracker
            hicl_tracker = HICLTracker(config=config, seqs=seqs, splits=splits)
            # Train the tracker
            hicl_tracker.train()
            print("####################")

        # Evaluate the performance of oracles and each epoch
        evaluate_mot17(tracker_path=osp.join(config.experiment_path, 'oracle'), split=config.cval_seqs, data_path=config.data_path, tracker_sub_folder=config.mot_sub_folder, output_sub_folder=config.mot_sub_folder)
        for e in range(1, config.num_epoch+1):
            evaluate_mot17(tracker_path=osp.join(config.experiment_path, 'Epoch' + str(e)), split=config.cval_seqs, data_path=config.data_path, tracker_sub_folder=config.mot_sub_folder, output_sub_folder=config.mot_sub_folder)

    # TESTING
    elif config.experiment_mode == 'test':
        # Get the splits for the experiment
        seqs, splits = get_seqs_from_splits(data_path=config.data_path, test_split=config.test_splits[0])

        # Initialize the tracker
        hicl_tracker = HICLTracker(config=config, seqs=seqs, splits=splits)

        # Load the pretrained model
        hicl_tracker.model = hicl_tracker.load_pretrained_model()

        # Track
        epoch_val_logs, epoc_val_logs_per_depth = hicl_tracker.track(dataset=hicl_tracker.test_dataset, output_path=osp.join(hicl_tracker.config.experiment_path, 'test'),
                                                                     mode='test',
                                                                     oracle=False)
               
        # Only works if you are testing on train or val data. Will fail in case of a test set
        evaluate_mot17(tracker_path=osp.join(hicl_tracker.config.experiment_path, 'test'), split=hicl_tracker.test_split,
                   data_path=hicl_tracker.config.data_path,
                   tracker_sub_folder=hicl_tracker.config.mot_sub_folder,
                   output_sub_folder=hicl_tracker.config.mot_sub_folder)
