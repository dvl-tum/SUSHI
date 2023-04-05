# Unifying Short and Long-Term Tracking with Graph Hierarchies :sushi:

Below, we provide the training and evaluation commands used to reproduce the results provided in the paper. For evaluating models, make sure you download [our checkpoints](https://drive.google.com/drive/folders/1cU7LeTAeKxS-nvxqrUdUstNpR-Wpp5NV?usp=share_link).


## MOT17

### MOT17 - public detections

For training, run:
```
RUN=mot17_public_train
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH=your_data_path
python scripts/main.py --experiment_mode train --cuda --train_splits mot17-train-all --val_splits mot17-train-all --run_id ${RUN} --interpolate_motion --linear_center_only --det_file aplift --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} --zero_nodes --reid_arch $REID_ARCH --edge_level_embed --save_cp
```

To obtain results on the test-set sequences run:
```
RUN=mot17_public_test
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH=your_data_path
PRETRAINED_MODEL_PATH=mot17public.pth
python scripts/main.py --experiment_mode test --cuda --test_splits mot17-test-all --run_id ${RUN} --interpolate_motion --linear_center_only --det_file aplift --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} --zero_nodes --reid_arch $REID_ARCH --edge_level_embed --save_cp --hicl_model_path ${PRETRAINED_MODEL_PATH}
```
Note that there are no ground-truth files. Evaluation is done in an online server.

### MOT17 - private detections
You just have to replace `--det_file aplift` with `--det_file byte065` in the previous two commands, as well as your checkpoint path.  Specifically, for training, run:
```
RUN=mot17_private_train
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH=your_data_path
python scripts/main.py --experiment_mode train --cuda --train_splits mot17-train-all --val_splits mot17-train-all --run_id ${RUN} --interpolate_motion --linear_center_only --det_file byte065 --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} --zero_nodes --reid_arch $REID_ARCH --edge_level_embed --save_cp
```

To obtain results on the test-set sequences run:
```
RUN=mot17_private_test
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH=your_data_path
PRETRAINED_MODEL_PATH=mot17private.pth
python scripts/main.py --experiment_mode test --cuda --test_splits mot17-test-all --run_id ${RUN} --interpolate_motion --linear_center_only --det_file byte065 --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} --zero_nodes --reid_arch $REID_ARCH --edge_level_embed --save_cp --hicl_model_path ${PRETRAINED_MODEL_PATH}
```
Note that there are no ground-truth files. Evaluation is done in an online server.

### MOT17 - cross-validation
For ablation studies, we perform 3-fold cross-validation on the MOT17 train sequences. To obtain cross-validation, you can run the following:
```
RUN=mot17_public_crossval
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH=your_data_path
for SPLIT in 1 2 3
do
python scripts/main.py --experiment_mode train --cuda --train_splits mot17-train-split${SPLIT}--val_splits mot17-val-split${SPLIT} --run_id ${RUN}_split${SPLIT} --interpolate_motion --linear_center_only --det_file aplift --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} --zero_nodes --reid_arch $REID_ARCH --edge_level_embed --save_cp
done
```
After training, you can combine the results from all three runs with
```
python scripts/combine_cval_scores.py --experiment_mode test --run_id ${RUN}
```

## MOT20
### MOT20 - public detections
For training, run:
```
RUN=mot20_public_train
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH=your_data_path
python scripts/main.py --experiment_mode train --cuda --train_splits mot20-train-all 
--val_splits mot20-val-split1 --run_id ${RUN} --interpolate_motion --linear_center_only --det_file aplift --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} --zero_nodes --reid_arch $REID_ARCH --edge_level_embed --save_cp --pruning_method geometry motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 motion_01
```

To obtain results on the test-set sequences run:
```
RUN=mot20_public_test
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH=your_data_path
PRETRAINED_MODEL_PATH=mot20public.pth
python scripts/main.py --experiment_mode test --cuda --test_splits mot20-test-all --run_id ${RUN} --interpolate_motion --linear_center_only --det_file aplift --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} --zero_nodes --reid_arch $REID_ARCH --edge_level_embed --save_cp --pruning_method geometry motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 --hicl_model_path ${PRETRAINED_MODEL_PATH} 
```
Note that there are no ground-truth files. Evaluation is done in an online server.

### MO20 - private detections
You just have to replace `--det_file aplift` with `--det_file byte065` in the previous two commands, as well as your checkpoint path.  Specifically, for training, run:
```
RUN=mot20_private_train
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH=your_data_path
python scripts/main.py --experiment_mode train --cuda --train_splits mot20-train-all 
--val_splits mot20-val-split1 --run_id ${RUN} --interpolate_motion --linear_center_only --det_file byte065 --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} --zero_nodes --reid_arch $REID_ARCH --edge_level_embed --save_cp --pruning_method geometry motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 motion_01
```

To obtain results on the test-set sequences run:
```
RUN=mot20_private_test
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH=your_data_path
PRETRAINED_MODEL_PATH=mot20private.pth
python scripts/main.py --experiment_mode test --cuda --test_splits mot20-test-all --run_id ${RUN} --interpolate_motion --linear_center_only --det_file byte065 --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} --zero_nodes --reid_arch $REID_ARCH --edge_level_embed --save_cp --pruning_method geometry motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 --hicl_model_path ${PRETRAINED_MODEL_PATH}
```
Note that there are no ground-truth files. Evaluation is done in an online server.

## DanceTrack
For training, run:
```
RUN=dancetrack_train
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH=your_data_path
python scripts/main.py --experiment_mode train --cuda --train_splits dancetrack-train-all --val_splits dancetrack-val-all --run_id ${RUN}_${REID_ARCH} --interpolate_motion --linear_center_only --det_file byte065 --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} --zero_nodes --reid_arch $REID_ARCH --edge_level_embed --save_cp --pruning_method geometry motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 --no_interpolate
```

To obtain results on the test sequences, run:
```
RUN=dancetrack_test
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH=your_data_path
PRETRAINED_MODEL_PATH=dancetrack.pth
python scripts/main.py --experiment_mode test --cuda --test_splits dancetrack-test-all --run_id ${RUN} --interpolate_motion --linear_center_only --det_file byte065 --data_path ${DATA_PATH} --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} --zero_nodes --reid_arch $REID_ARCH --edge_level_embed --save_cp --pruning_method geometry motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 --hicl_model_path ${PRETRAINED_MODEL_PATH} --no_interpolate
```
Note that there are no ground-truth files. Evaluation is done in an online server.

## BDD
BDD dataset contains multiple classes and we have a naive extension of our model in which we simply apply our GNNs across all classes. 

1. Switch to branch `bdd`

2. BDD folder format is different than MOTChallenge and DanceTrack format. Hence, we restructure the BDD dataset into MOTChallenge format. Modify `scripts/bdd_to_motcha.py` for your current folder structure to accomplish that

3. Modify bdd_path variable in `src/data/seq_processer.py` according to your new folder structure

For training, run:
```
RUN=bdd_train
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH=your_data_path
python scripts/main.py --experiment_mode train --cuda --train_splits bdd-train-all --val_splits bdd-val-all --frames_per_graph 320 --top_k_nns 12 --hicl_depth 7 --frames_per_level 5 10 20 40 80 160 320 --depth_pretrain_iteration 500 --run_id ${RUN}_${REID_ARCH} --motion_pred_length 5 10 20 40 80 160 --motion_max_length 5 10 20 40 60 100 --interpolate_motion --linear_center_only --pruning_method geometry motion_005 motion_005 motion_005 motion_005 motion_005 motion_005 --det_file dets_bdd_byte --data_path ${DATA_PATH} --mpn_use_reid_edge 1 1 1 1 1 1 1 --mpn_use_pos_edge 1 1 1 1 1 1 1 --mpn_use_motion 0 1 1 1 1 1 1 --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} --zero_nodes --share_weights all_but_first --reid_arch $REID_ARCH --edge_level_embed --save_cp --no_interpolate --no_fp_loss --multiclass
```

To obtain results on the test sequences, run:
```
RUN=bdd_val
REID_ARCH='fastreid_msmt_BOT_R50_ibn'
DATA_PATH=your_data_path
PRETRAINED_MODEL_PATH=bdd.pth


python scripts/main.py --experiment_mode test --cuda --test_splits bdd-val-all --frames_per_graph 320 --top_k_nns 12 --hicl_depth 7 --frames_per_level 5 10 20 40 80 160 320 --depth_pretrain_iteration 500 --run_id ${RUN}_${REID_ARCH} --motion_pred_length 5 10 20 40 80 160 --motion_max_length 5 10 20 40 60 100 --interpolate_motion --linear_center_only --pruning_method geometry motion_005 motion_005 motion_005 motion_005 motion_005 motion_005 --det_file dets_bdd_byte --data_path ${DATA_PATH} --mpn_use_reid_edge 1 1 1 1 1 1 1 --mpn_use_pos_edge 1 1 1 1 1 1 1 --mpn_use_motion 0 1 1 1 1 1 1 --reid_embeddings_dir reid_${REID_ARCH} --node_embeddings_dir node_${REID_ARCH} --zero_nodes --share_weights all_but_first --reid_arch $REID_ARCH --edge_level_embed --save_cp --no_interpolate --no_fp_loss --multiclass --hicl_model_path ${PRETRAINED_MODEL_PATH}
```

Note: We are using TrackEval for the evaluation of BDD but we noticed some differences in the tracking scores between the official evaluation server and TrackEval. We suggest using the official BDD evaluation server for BDD tracking scores. You can use `scripts/bdd_submission.py` to convert your tracking output files to BDD submission server format. 
