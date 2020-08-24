#!/bin/sh
python3 /home/jetson/tensorpack/examples/FasterRCNN/predict.py \
  --predict /home/jetson/Documents/online_service/static/images/original.jpg \
  --load /home/jetson/Documents/trained_model/51epoch/checkpoint \
  --config \
    MODE_MASK=False
