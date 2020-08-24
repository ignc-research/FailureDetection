# Industrial Failure Detection with Neural Networks



## Overview

Train Faster R-CNN, Cascase R-CNN, and RetinaNet models on Tensorpack and MMDetection with thermal images for failure detection.

1. Prepare dataset

   1. Extract thermal(There’re 115 films in total, 250 images contained in each of them.)
   2. Data filtering: bluring, and adjacent similiar images.
   3. Labeling: **[LabelImg](https://github.com/tzutalin/labelImg)**, save annotaions with **Pascal VOC** data format. Used labels refer to 'data_engineering'.
   4. Test set: pick out 10(including all labels) films for final test.
   5. Augmentation: find the needed scripts in the folder 'augmentation'.

2. Setup training frameworks

   1. Install [MMDetection](https://github.com/open-mmlab/mmdetection)

      Relatted Adaptation: data foler; class names; used detectior; pre-trained model; results folder.

   2. Install [Tensorpack](https://github.com/tensorpack/tensorpack)

      Relatted Adaptation: data foler; class names; used detectior; pre-trained model; results folder.

3. Start training

   1. Scripts provided. For ease of changing detectors, use some shell scripts.
   2. Monitoring the process with tensorboard(Tensorpack) and built-in tools(MMDetection).

4. Evaluation/Test (scripts provided).

5. Deployment:

   On platform Nvidia Jetson AGX Xavier(Nvidia Xavier).