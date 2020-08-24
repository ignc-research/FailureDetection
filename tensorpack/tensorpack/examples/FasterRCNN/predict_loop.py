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
from datetime import datetime
from os import system

import cv2
import numpy as np
import tensorflow as tf
import tqdm
from smb.SMBConnection import SMBConnection

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

#**server_file_list** needs to be initialized in main()
def smb_conn_sync():
    '''
    smb connection: copy files from server to local
    To adapt:  share_folder, user_name, pwd, server_name, server_ip
    '''
    share_folder = 'myshare' # 'SAPtransfer'
    user_name = 'mytest' # 'thermoanlage'
    pwd = 'mytest' # 'thermo2010$'
    server_name = 'xp_server' # 'SIMATIC'
    server_ip = '192.168.0.10' # '192.168.5.2'
    
    global server_folder_list
    global source_folder_list
    local_dir = source_folder_path
    
    smb_conn = SMBConnection(user_name, pwd, 'jetson-desktop', server_name, use_ntlm_v2 = False)
    assert smb_conn.connect(server_ip, 139)
    # print(datetime.now(), 'SMB connection established....')
    
    filelist = smb_conn.listPath(share_folder, '/')

    f_names = []
    for i in range(len(filelist)): # get file name list from file path list
        if filelist[i].filename.endswith('.jpg'): # if 'TM' in filelist[i].filename:
            filename = filelist[i].filename
            f_names.append(filename)

    if server_folder_list == [0]: # server list is NOT iniatialized yet
        server_folder_list = f_names
        return
    for file in f_names: # copy new added files to local source
        if file not in server_folder_list:
            with open(local_dir + '/' + file, 'wb') as fp:
                print('smb.....copy:', file)
                smb_conn.retrieveFile(share_folder,'/'+ file, fp )
                source_folder_list.append(file)

    server_folder_list = f_names # update server_folder_list
    smb_conn.close() # relase resources from local and server

def do_compare():
    # compare *source folder* and predicted_image_queue
    # do prediction on new img
    global predicted_image_queue
    print(datetime.now().strftime("%H:%M:%S %d/%m/%Y"), 'Service running....')
    for file in source_folder_list:
        if file not in predicted_image_queue:
            print('found new file in source folder..........')
            predicted_image_queue.append(file)
            sort_result(result_folder_path) # change old files names for new img
            do_predict(predictor, source_folder_path + '/' + file)

def sort_result(path):
    result_images = os.listdir(path)
    result_images = sorted(result_images, reverse=True)
    #print(result_images)
    # rename file, plus one to every name, e.g change *0.png* to *1.png*
    for file in result_images:
        old_file_path = path + '/' + file
        file_name = os.path.splitext(file)[0]

        if (int(file_name)+1) == max_num_results: 
            os.remove(path + '/' + file_name + '.png')
            continue

        new_file_name = str(int(file_name)+1)
        new_file_path = path + '/' + new_file_name + '.png'
        print(old_file_path, new_file_path)
        os.rename(old_file_path, new_file_path)

def do_predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image(img, pred_func)
    final = draw_final_outputs(img, results)

    # add green rectangle arround original picture that with failure
    #height, width, channels = img.shape
    #cv2.rectangle(img, (0, 0), (width, height), color=(100, 220, 80), thickness=5)

    #viz = np.concatenate((img, final), axis=1)
    #cv2.imwrite(result_folder_path+"/0.png", viz)
    cv2.imwrite(result_folder_path+"/0.png", final)
    logger.info("Inference output written to 0.png")

def cleanup_folders(paths):
    '''
    delete && re-create all related folders
    '''
    for folder in paths:
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder)

def init_predictor():
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
    
    return predictor

def print_info():
    system('clear') # clear terminal ouput
    print(datetime.now().strftime("%H:%M:%S %d/%m/%Y"), 'Service running....')
    print()
    print('***Only the newest *5* images will be displayed in browser!')
    print()
    print()
    print('Put more new images in server shared folder to test.')

if __name__ == '__main__':
    result_folder_path = '/home/jetson/xavier_page/static/images'
    source_folder_path = '/home/jetson/xavier_page/source_folder'
    cleanup_folders([result_folder_path, source_folder_path, ]) # delete contents inside
    
    source_folder_list = os.listdir(source_folder_path)
    server_folder_list = [0] # 0 menas NOT initialized, check and modified in smb_conn_sync()

    max_num_results = 10 # max num of kept prediction results
    predicted_image_queue = [] # track predicted img in order

    predictor = init_predictor()
    do_predict(predictor, '/home/jetson/tensorpack/examples/FasterRCNN/test.jpg')
    webbrowser.open_new('file:///home/jetson/xavier_page/index.html') # browser show images in reuslt folder

    while True:
        time.sleep(1)
        smb_conn_sync() # copy new images from server to local source folder
        do_compare()    # do prediction on new images in source folder
        #print_info()    # print some hints on terminal


