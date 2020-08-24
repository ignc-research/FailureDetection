# Industrial Failure Detection with Neural Networks



## Overview

After Training is done, trained model can be used for detection, either directly on the GPU Server with the same training setup, or on other embeeded system like Nvidia Jetson AGX Xavier board.

- On GPU Server: with the same framework, inference scripts are provided.
- On Xavier: Need to install Tensorpack, following the Nvidia developer website. The others are the same like on GPU Server.



# Deploy Tensorpack on Jetson Xavier

From here, we turn from host traiing pc to the Nvidia Xavier board. To use Xavier for reference, basically we have two ways, **one** is install whole tensorpack on Xavier, **the other** is convert our model to ONNX for TensorRT. More details about TensorRT can be found on Nvidia website. Here we’ll install complete tensorpack for easy of deployment. Later we 'll test TensorRT for it’s potential benifits.

As we only use Xavier for reference, the powerful but complicated prediction commands are not suitable here. For easy of use, we'll deploy the whole system as a **web service** with the help of **Flask**. In the following steps, first we install tensorpack on Xavier and make modifications like we did on Host. Then install Flask, and prepare flask app script for web service deployment.

**Reminds**: Nvidia Jetson modules use **aarch64** architecture, so there might come with compatibility problems when installing packages. Here we mainly use the officially provided sources to avoid that. Besides, try "**sudo -H --user** install pkgs" to avoid permission problems.

Setup Xavier: **Install JetPack 4.2**, following official guides. After install, we have python3.6 and pip3 installed by default, here we make alias for python and pip, as we **only use python3 and pip3**, open /home/jetson/.bashrc, add:

```shell
alias python="python3.6"
alias pip="pip3"
```

Following the tensorpack install guide for dependencies, but for tensorflow, please follow [Official TensorFlow for Jetson AGX Xavier](https://devtalk.nvidia.com/default/topic/1042125/jetson-agx-xavier/official-tensorflow-for-jetson-agx-xavier/) and [TensorFlow For Jetson Platform](http://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html). The tested version combination is "**tensorflow-gpu==1.14.0+nv19.10**". The related commnads:

```shell
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev
sudo pip3 install -U numpy grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast h5py astor termcolor protobuf keras-applications keras-preprocessing wrapt google-pasta
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu==1.14.0+nv19.10
```

Here are some encountered problems with pip install, these can solved by apt-get or with "sudo -H --user":

```shell
sudo -H pip install cython --user
sudo apt-get install python3-matplotlib
sudo -H apt-get install python3-scipy
sudo -H pip install pycocotools
```

Download && install tensorpack:

```shell
git clone https://github.com/tensorpack/tensorpack.git
pip install --upgrade git+https://github.com/tensorpack/tensorpack.git --user
```

Modify related parameters in coco.py and config.py like above. 

Upload trained model files and test images:

<img src="../../../../Documents/Typora_base_folder/Screenshot%25202019-12-10%2520at%252018.25.14.png" style="zoom:50%;" />

Now we can test our tensorpack installation with script:

```shell
#!/bin/sh
python3 /home/jetson/tensorpack/examples/FasterRCNN/predict.py \
    --predict path/to/test/thermal_image.jpg \
    --load /home/jetson/Documents/trained_model/51epoch/checkpoint \
    --config \
      MODE_MASK=False
```

If "output.png" file is generated, that means the tensorpack is successfully installed. Now we start to deploy our web API to provide reference service.

Install Flask:

```
sudo -H pip install flask --user
sudo -H pip install flask-wtf
```

Here we use Flask to provide web service, and refer to "predict.py" to build "do_prediction()" for reference function:

```
# tensorpack/examples/FasterRCNN/xavier_flask_predict_app.py
# this script is a simplified version of fasterRCNN/predict.py script
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

app = Flask(__name__) # create flask starting fun

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

            # add timestamp to stop Browser from using cached image
            timestamp = str(time.time())
            result_img = "static/images/output.png" + "?" + timestamp  # outputed image

        photo_form = PhotoForm(request.form)
        return render_template('upload.html', photo_form=photo_form, result=result_img, color='red')

def do_predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image(img, pred_func) # get error from this
    final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)
    cv2.imwrite("static/images/output.png", viz)
    logger.info(
        "Inference output for {} written to output.png".format(input_file))

if __name__ == '__main__':
    register_coco(cfg.DATA.BASEDIR)
    MODEL = ResNetFPNModel()
    finalize_configs(is_training=False)

    predcfg = PredictConfig(
        model=MODEL,
        session_init=SmartInit("/home/jetson/Documents/trained_model/51epoch/checkpoint"),
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1])

    predictor = OfflinePredictor(predcfg)
    #do_predict(predictor, 'test.jpg')

    app.run(host='192.168.117.56', port=5000, debug=False)
```

In above script, we init a global "predictor" inside "if __name__ == '__main__'", thus we can keep the tensorpack and model in memory, no need to re-load every thing for next prediction. The related files should be organized like below:

```
tensorpack/examples/FasterRCNN/
	***
	static/images/
	templates/
		formhelpers.html
		upload.html
	***
	predict.py
	train.py
	***
	xavier_flask_predict_app.py
```

```javascript
# formhelpers.html
# -*- coding: utf-8 -*-

{% macro render_field(field) %}
  <p>{{ field.label }}</p>
  <p>{{ field(**kwargs) | safe }}</p>
  {% if field.errors %}
    <ul class=errors>
    {% for error in field.errors %}
      <li>{{ error }}</li>
    {% endfor %}
    </ul>
  {% endif %}
{% endmacro %}
```

```html
# upload.html
<!doctype html>
<html>
  <head>
    <meta http-equiv="Content-type" content="text/html; charset=utf-8">
    <title>Object Detection API</title>
    <style>
      .green {border: solid greenyellow 5px}
      .red {border: solid red 5px}
      #align {border: solid green 5px}
      #left, #right {
        display: inline-block;
        margin: 10px
      }
    </style>
    <script>
      function WaitDisplay() {
        target = document.getElementById("result");
        target.style.display = "none";
        target = document.getElementById("loading");
        target.style.display = "";
        setTimeout(function () {
          document.getElementById("upload").submit();
        }, 100);
      }
    </script>
  </head>

  <body>
    {% from "formhelpers.html" import render_field %}
    
    <h1>Failure Detection API</h1>
    <h3>Upload a thermal photo file.</h3>

    <div>
      <form id="upload" method=post action={{ url_for('post') }} enctype="multipart/form-data">
        {{ render_field(photo_form.input_photo) }}
        <p><input type="button" onclick="WaitDisplay();" value="Upload"></p>
      </form>
    </div>
    <hr>
    <div id="result">
      {% if result|length > 0 %}
        <img class="{{ color }}" id="photo" src="{{ result }}" />
      {% endif %}
    </div>
    <div id="loading" style="display:none">
      <h2>Detecting failure...</h2>
    </div>

  </body>
</html>
```

**To start the Flask service**, open terminal, go to tensorpack/examples/FasterRCNN/:

```
# Need to specify the Xavier ip inside the script, default port number is 5000
# app.run(host='192.168.117.56', port=80, debug=False)

sudo python xavier_flask_predict_app.py # needs sudo for port 80
```

Then use browser on any desktop which can access the specified ip address:

<img src="/Users/fxl/Documents/Typora_base_folder/Screenshot%25202019-12-10%2520at%252018.41.16.png" style="zoom:60%;" />

## More about Nvidia Xavier

Except directly using Tensorpack, it offers TensorRT for better performance. Here are some research.

```
- [ ] More modification or just delet
```

##

Conversion of our model into a format which can be deployed on the nvidia board.

**Xavier provides following choices, we can focus on ONNX and UFF parsers.**

1. ONNX parser; 
2. TensorFlow/UFF parser; 
3. ~~Caffe parser;~~
4. ~~C++ and Python APIs for implementing the most common deep learning layers.~~

Three directions:

1. Install pytorch/mmdetection on Xavier won’t work: archh64 architecture don’t support certain dependencies like opencv-python for mmdetection
2. Convert trained pytorch model to support formats: **ONNX** or **UFF**
3. Train new model with Tensorflow implementation and export UFF model

The following parts are solutions tried and the encountered problems:

1. Convert faster-rcnn pytorch to ONNX

   1. pytorch official method:

      Pytorch github page closed answer: [Cannot export fasterrcnn/keypointrcnn/fasterrcnn to ONNX.       #27969](https://github.com/pytorch/pytorch/issues/27969)

   2. ONNX official method:

      https://github.com/onnx/tutorials#converting-to-onnx-format

      no builtin faster-rcnn support;

      didn’t find ready to use repo

   3. github user method:

      https://michhar.github.io/convert-pytorch-onnx/#a-pre-trained-model-from-torchvision

      didn’t try, it’s still needed to touch the graph

2. Convert pytorch to TensorRT

   https://github.com/modricwang/Pytorch-Model-to-TensorRT

   didn’t try this, as the repo requires CUDA == 9.0 CUDNN == 7.3.1 TensorRT == 4.0.2.6.

3. TensorRT builtin sample Faster RCNN won’t work, it can’t be fine tuned.

   Nvidia forum: https://devtalk.nvidia.com/default/topic/1028409/jetson-tx2/how-can-i-finetune-the-tensorrt-faster-rcnn-sample-/ 「The model is trained by the faster R-CNN author and is slightly modified in RPN and ROIPooling for TensorRT plugin interface. Faster R-CNN training requires author's custom Caffe branch. Check here for more information: [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)」

4. Train Keras/Tensorflow implementation, export UFF model

   https://github.com/raghavb/frcnn-from-scratch-with-keras

   **Training the Keras model** :

   ***Problem\***: I tried training the model on Google Colabs but it was extremely slow and crashed after 1 epoch. 

   **Solution**: Training again on Google Colabs, trying the training on local machine or Training on the server in the university.

   **Task: Converting the Keras model to TensorRT:**

   In order to make the model compatible with Nvidia Xavier, the Keras model needs to be converted into TensorRT which can be deployed to Jetson

   **Approach**: https://github.com/ardianumam/Tensorflow-TensorRT
    	https://github.com/NVIDIA-AI-IOT/tf_trt_models

   



As our understanding till now, the conversion replys on the support from pytorch itself, as the discuss mentioned here  [Cannot export fasterrcnn/keypointrcnn/fasterrcnn to ONNX.       #27969](https://github.com/pytorch/pytorch/issues/27969). So to proceed with a Keras/tensorflow implementation would be more promising:

1. the solution Charul is working on.
2. Tensorpack: problem, dataset needs to be converted to coco.

