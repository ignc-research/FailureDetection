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
from flask import Flask, redirect, render_template, request, url_for
from flask_wtf.file import FileField
from PIL import Image
from werkzeug.datastructures import CombinedMultiDict
from wtforms import Form, ValidationError

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


# create flask starting fun
app = Flask(__name__)

# do_predict(predictor, image_file)
def do_predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image(img, pred_func) # get error from this

    img_name = ntpath.basename(input_file)
    final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)
    save_path = "/home/jetson/Documents/result/"+model_num+"/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    cv2.imwrite(save_path+img_name, viz)
    logger.info(
        "Inference output for {} written to {}".format(input_file, save_path))

if __name__ == '__main__':
    register_coco(cfg.DATA.BASEDIR)
    MODEL = ResNetFPNModel()
    finalize_configs(is_training=False)

    predcfg = PredictConfig(
        model=MODEL,
        session_init=SmartInit("/home/jetson/Documents/598000/checkpoint"),
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1])

    predictor = OfflinePredictor(predcfg)

    model_num = "507500"
    f1 = []
    for (dirpath, dirnames, filenames) in walk("/home/jetson/Documents/16_Dec"):
        f1.extend(filenames)
        break
    for img in f1:
        img = "/home/jetson/Documents/16_Dec/" + img
        do_predict(predictor, img)
    
    print("Done!")
