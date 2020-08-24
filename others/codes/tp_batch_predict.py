# this script modified from built-in predict.py script
# # original codes refer to fasterRCNN/predict.py
import argparse
import itertools
import os
import os.path
import shutil
import socket
import subprocess  # for calling shell script
import sys
import tempfile
import time

import cv2
import numpy as np
import tensorflow as tf
import tqdm

import tensorpack.utils.viz as tpviz
from config import config as cfg
from config import finalize_configs
from data import get_eval_dataflow, get_train_dataflow
from dataset import DatasetRegistry, register_balloon, register_coco
from eval import DetectionResult, multithread_predict_dataflow, predict_image
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel
from tensorpack.predict import (MultiTowerOfflinePredictor, OfflinePredictor,
                                PredictConfig)
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.tfutils.export import ModelExporter
from tensorpack.utils import fs, logger
from viz import (draw_annotation, draw_final_outputs,
                 draw_final_outputs_blackwhite, draw_predictions,
                 draw_proposal_recall)

from os import walk
from shutil import copyfile
import time
import ntpath

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# do_predict(predictor, image_file)
def do_predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image(img, pred_func) # get error from this

    img_name = ntpath.basename(input_file)
    final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    cv2.imwrite(result_folder + img_name, viz)
    logger.info("Inference output for {} written to {}".format(input_file, result_folder))

if __name__ == '__main__':
    register_coco(cfg.DATA.BASEDIR)
    MODEL = ResNetFPNModel()
    finalize_configs(is_training=False)

    predcfg = PredictConfig(
        model=MODEL,
        session_init=SmartInit("data/train_log_bam/27.01.2020_bam_old_backup/checkpoint"),
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1])

    predictor = OfflinePredictor(predcfg)
    
    source_folder = "data/forTest_BAM/old/"
    result_folder = source_folder + "../" +"result_old_50000/"

    f1 = []
    for (dirpath, dirnames, filenames) in walk(source_folder):
        f1.extend(filenames)

    for img in f1:
        img = source_folder + img
        do_predict(predictor, img)
    
    print("Done!")
