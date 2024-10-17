#!/usr/bin/env bash

set -x

FILE_NAME=$(basename $0)
EXP_DIR="/home/aditya/snaglist_training_plain_detr"
PY_ARGS=${@:1}

python -u main.py \
    --coco_path /home/aditya/snaglist_dataset_apr9 \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --mixed_selection \
    --look_forward_twice \
    --num_queries_one2one 300 \
    --num_queries_one2many 1500 \
    --k_one2many 6 \
    --lambda_one2many 1.0 \
    --dropout 0.3 \
    --norm_type pre_norm \
    --backbone swin_v2_small_window12to16_2global \
    --drop_path_rate 0.1 \
    --upsample_backbone_output \
    --upsample_stride 16 \
    --num_feature_levels 1 \
    --decoder_type global_rpe_decomp \
    --decoder_rpe_type linear \
    --proposal_feature_levels 4 \
    --proposal_in_stride 16 \
    --pretrained_backbone_path /home/aditya/Plain-DETR/map50_plain_detr.pth \
    --epochs 12 \
    --lr_drop 11 \
    --warmup 1000 \
    --lr 2e-4 \
    --use_layerwise_decay \
    --lr_decay_rate 0.9 \
    --weight_decay 0.05 \
    --wd_norm_mult 0.0 \
    --reparam \
    --set_cost_bbox 1.0 \
    --bbox_loss_coef 1.0 \
    --position_embedding sine_unnorm \
    --batch_size 1 \
    ${PY_ARGS}
