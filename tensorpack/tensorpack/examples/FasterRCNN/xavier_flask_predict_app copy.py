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

# checkpoint_file = '/home/jetson/Documents/trained_model/51epoch/checkpoint'
# static_folder = '/home/jetson/tensorpack/examples/FasterRCNN/static'
# upload_image_folder = '/home/jetson/tensorpack/examples/FasterRCNN/static/images'

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
            image.save(image_path + "original.jpg") # save uploaded images to server
            # delete old output.png
            if (os.path.isfile(image_path + 'output.png')):
                os.remove(image_path + 'output.png')

            do_predict(predictor, image) # result image 'output.png' saved to /static/images/output.png
            #result_data = detect_objects(img)
            #result= result_data['result']

            while not os.path.isfile(image_path + 'output.png'):
                print('prediction failed')
                return

            # add timestamp to image url to make Browser refresh without using cached image
            timestamp = str(time.time())
            result_img = image_path + "output.png" + "?" + timestamp  # predict output image
            img_timestamp = image_path + "original.jpg?" + timestamp  # uploaded original image

        photo_form = PhotoForm(request.form)

        return render_template('upload.html', photo_form=photo_form, result=result_img, color='red', original=img_timestamp)


# do_predict(predictor, image_file)
def do_predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image(img, pred_func)
    if cfg.MODE_MASK:
        final = draw_final_outputs_blackwhite(img, results)
    else:
        final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)
    cv2.imwrite(image_path + "output.png", viz)
    logger.info(
        "Inference output for {} written to output.png".format(input_file))
    # tpviz.interactive_imshow(viz)


if __name__ == '__main__':
    # init model predictor with training configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--load', help='load a model for evaluation.', required=True)
    parser.add_argument('--visualize', action='store_true',
                        help='visualize intermediate results')
    parser.add_argument('--evaluate', help="Run evaluation. "
                                           "This argument is the path to the output json evaluation file")
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file", nargs='+')
    parser.add_argument('--benchmark', action='store_true',
                        help="Benchmark the speed of the model + postprocessing")
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')
    parser.add_argument('--output-pb', help='Save a model to .pb')
    parser.add_argument('--output-serving',
                        help='Save a model to serving file')

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)
    register_coco(cfg.DATA.BASEDIR)  # register COCO datasets
    register_balloon(cfg.DATA.BASEDIR)

    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    if not tf.test.is_gpu_available():
        from tensorflow.python.framework import test_util
        assert get_tf_version_tuple() >= (1, 7) and test_util.IsMklEnabled(), \
            "Inference requires either GPU support or MKL support!"
    assert args.load
    finalize_configs(is_training=False)

    if args.predict or args.visualize:
        cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

    # 'args.visualize' removed
    predcfg = PredictConfig(
        model=MODEL,
        session_init=SmartInit(args.load),
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1])

    predictor = OfflinePredictor(predcfg)
    #do_predict(predictor, image_file)
    
    # get local host ip
    #ip_str = socket.gethostbyname(socket.gethostname())
    image_path = "./templates/static/images/"
    app.run(host='192.168.117.56', port=5000, debug=False)
