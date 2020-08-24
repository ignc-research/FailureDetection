#!/bin/sh
python train.py \
    --logdir data/train_log/17122019 \
    --config \
        DATA.BASEDIR=data/COCO/DIR \
        BACKBONE.WEIGHTS=data/ImageNet-R50-AlignPadding.npz \
        MODE_MASK=False \
        DATA.NUM_WORKERS=4 \
        TRAIN.CHECKPOINT_PERIOD=5 \
        TRAIN.EVAL_PERIOD=5