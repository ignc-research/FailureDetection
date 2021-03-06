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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#checkpoint_file = '/home/jetson/Documents/trained_model/51epoch/checkpoint'

# create flask starting fun
app = Flask(__name__)

content_types = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png'}
extensions = sorted(content_types.keys())

def is_image():
    def _is_image(form, field):
        if not field.data:
            raise ValidationError()
        elif field.data.filename.split('.')[-1].lower() not in extensions:
            raise ValidationError()

    return _is_image

class PhotoForm(Form):
    input_photo = FileField(
        'File extension should be: %s (case-insensitive)' % ', '.join(extensions),
        validators=[is_image()])

@app.route('/')
def upload():
    photo_form = PhotoForm(request.form)
    return render_template('upload.html', photo_form=photo_form, result={})

@app.route('/post', methods=['GET', 'POST'])
def post():
    form = PhotoForm(CombinedMultiDict((request.files, request.form)))
    if request.method == 'POST' and form.validate():
        with tempfile.NamedTemporaryFile() as temp:
            form.input_photo.data.save(temp)
            temp.flush()

            image = Image.open(temp.name).convert('RGB') # get uploaded images
            image.save("static/images/original.jpg") # save uploaded images to server

            do_predict(predictor, "static/images/original.jpg") # result image 'output.png' saved to /static/images/output.png

            while not os.path.isfile("static/images/output.png"):
                print('prediction failed')
                return

            # add timestamp to image url to make Browser refresh without using cached image
            timestamp = str(time.time())
            result_img = "static/images/output.png" + "?" + timestamp  # predict output image

        photo_form = PhotoForm(request.form)

        return render_template('upload.html', photo_form=photo_form, result=result_img, color='red')

# do_predict(predictor, image_file)
def do_predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image(img, pred_func) # get error from this
    final = draw_final_outputs(img, results)

    # add green rectangle arround original picture that with failure
    height, width, channels = img.shape
    cv2.rectangle(img, (0, 0), (width, height), color=(0, 255, 0), thickness=3)
    
    viz = np.concatenate((img, final), axis=1)
    cv2.imwrite("static/images/output.png", viz)
    logger.info("Inference output written to output.png")

if __name__ == '__main__':
    register_coco(cfg.DATA.BASEDIR)
    MODEL = ResNetFPNModel()
    finalize_configs(is_training=False)

    predcfg = PredictConfig(
        model=MODEL,
        session_init=SmartInit("/home/jetson/Documents/trained_model/4490/checkpoint"),
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1])

    predictor = OfflinePredictor(predcfg)
    #do_predict(predictor, 'test.jpg')

    app.run(host='192.168.117.90', port=5000, debug=False) # port 80 needs sudo permission
