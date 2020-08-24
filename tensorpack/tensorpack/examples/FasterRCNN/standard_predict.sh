#!/bin/sh
python3 predict.py \
  --predict test.jpg \
  --load /home/jetson/Documents/trained_model/51epoch/checkpoint \
  --config \
    MODE_MASK=False
