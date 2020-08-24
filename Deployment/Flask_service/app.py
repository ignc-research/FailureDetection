import os
import os.path
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import itertools
import numpy as np
from PIL import Image

import shutil
import tensorflow as tf
import cv2
import tqdm
import sys
import tempfile

from flask import Flask
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from flask_wtf.file import FileField

from werkzeug.datastructures import CombinedMultiDict
from wtforms import Form
from wtforms import ValidationError
import time

import subprocess # for calling shell script

checkpoint_file = '/home/jetson/Documents/trained_model/51epoch/checkpoint'

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
      # print("............test one")
      image = Image.open(temp.name).convert('RGB')
      image.save("./static/images/" + "original.jpg")
      # delete old output.png
      if (os.path.isfile('/home/jetson/tensorpack/examples/FasterRCNN/output.png')):
        os.remove('/home/jetson/tensorpack/examples/FasterRCNN/output.png')

      #result_data = detect_objects(img)
      #result= result_data['result']

      while not os.path.isfile('/home/jetson/Documents/online_service/output.png'):
          subprocess.call(['/home/jetson/Documents/online_service/web_predict.sh'])

      result = cv2.imread('/home/jetson/tensorpack/examples/FasterRCNN/output.png',)

      timestamp = str(time.time()) # add timestamp to image url to make Browser refresh without using cached image
      result_img = result + "?" + timestamp # predict output image
      img_timestamp = "./static/images/original.jpg?" + timestamp # uploaded original image

    photo_form = PhotoForm(request.form)

    return render_template('upload.html',photo_form=photo_form, result=result_img, color='red', original=img_timestamp)



if __name__ == '__main__':
  app.run(host='192.168.117.57', port=5000, debug=True)
