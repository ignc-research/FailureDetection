# this script modified from built-in predict.py script
# # original codes refer to fasterRCNN/predict.py
import threading
from threading import Timer
from shutil import copyfile
from os import remove, walk
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
import webbrowser

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def do_compare():
    print('Beacon ', num, 'Service Online, check in browser.!', f2, len(f2))

    # f1, f2 global var
    for (dirpath, dirnames, filenames) in walk(source_folder_path):
        for file in filenames:
            if target in file and target not in f1:
                f1.append(file)
    '''
    for (dirpath, dirnames, filenames) in walk(target_folder_path):
        f2.extend(filenames)
    '''
    for file in f1:
        if file not in f2:
            if file not in f2_origin:
                print(file, 'gefunden and kopiert')
                f2.append(file)
                file_path = target_folder_path+'/'+file
                copyfile(source_folder_path+'/'+file, target_folder_path+'/'+file)

                if len(f2) == 6:  # keep at most 5 images
                    f2_origin.append(f2[0])
                    os.remove(target_folder_path + '/' + f2[0])
                    f2.pop(0)
                    
                sort_result()
                do_predict(predictor, file_path)

def sort_result():
    target = '/home/jetson/xavier_page/static/images'

    result_images = []
    for (dirpath, dirnames, filenames) in walk(target):
        result_images.extend(filenames)
    result_images = sorted(result_images, reverse=True)

    for file in result_images:
        old_file_path = target + '/' + file
        file_name = os.path.splitext(file)[0]
        new_file_name = str(int(file_name)+1)
        new_file_path =  target + '/' + new_file_name + '.png'
        print(old_file_path, new_file_path)
        os.rename(old_file_path, new_file_path)
        

def do_predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image(img, pred_func)
    final = draw_final_outputs(img, results)

    # add green rectangle arround original picture that with failure
    height, width, channels = img.shape
    cv2.rectangle(img, (0, 0), (width, height),
                  color=(100, 220, 80), thickness=5)

    viz = np.concatenate((img, final), axis=1)
    cv2.imwrite(result_folder_path+"/0.png", viz)
    logger.info("Inference output written to 0.png")


def cleanup_folders():
    '''
    delete && re-create all related folders
    '''
    paths = [
        "COCO/DIR/ImageSets/Main",
        "COCO/DIR/annotations",
        "COCO/DIR/train2019",
        "COCO/DIR/val2019",
    ]
    for folder in ['COCO']:
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)

    for folder in paths:
        if not os.path.exists(folder):
            os.makedirs(folder)



'''
def retrieve_local_share():
    pass
'''


if __name__ == '__main__':
    result_folder_path = '/home/jetson/xavier_page/static/images'
    source_folder_path = '/home/jetson/xavier_page/source_folder'
    # if retrieve from local network folder
    # source_folder_path = retrieve_local_share()
    target_folder_path = '/home/jetson/xavier_page/target_folder'
    
    # clean up old images in target_folder_path && result_folder_path
    if os.path.exists(result_folder_path):
        shutil.rmtree(result_folder_path, ignore_errors=True)
        os.makedirs(result_folder_path)
    if os.path.exists(target_folder_path):
        shutil.rmtree(target_folder_path, ignore_errors=True)
        os.makedirs(target_folder_path)
    
    target = 'TM'
    f1, f2, f2_origin = [], [], []

    for (dirpath, dirnames, filenames) in walk(target_folder_path):
        for file in filenames:
            os.remove(target_folder_path + '/' + file)
    for (dirpath, dirnames, filenames) in walk(source_folder_path):
        for file in filenames:
            if target in file:
                f1.append(file)
                f2_origin.append(file)


    register_coco(cfg.DATA.BASEDIR)
    MODEL = ResNetFPNModel()
    finalize_configs(is_training=False)

    predcfg = PredictConfig(
        model=MODEL,
        #session_init=SmartInit("/home/jetson/Documents/trained_model/500000_17/checkpoint"),
        session_init=SmartInit("/home/jetson/Documents/trained_model/255000_04.01/checkpoint"),
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1])

    predictor = OfflinePredictor(predcfg)
    do_predict(
        predictor, "/home/jetson/tensorpack/examples/FasterRCNN/static/images/original.jpg")

    webbrowser.open_new('file:///home/jetson/xavier_page/index.html')

    num = 0
    while True:
        time.sleep(1)
        do_compare()
        num += 1


