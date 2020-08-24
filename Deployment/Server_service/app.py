import sys
import tempfile
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#from decorator import requires_auth
from flask import Flask
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from flask_wtf.file import FileField
import numpy as np
from PIL import Image
import cv2
import mmcv
from mmcv.runner import load_checkpoint
import mmcv.visualization.image as mmcv_image
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, init_detector

from werkzeug.datastructures import CombinedMultiDict
from wtforms import Form
from wtforms import ValidationError
import time

checkpoint_file = './models/epoch_14.pth'
config_fname = './models/faster_rcnn_r50_fpn_1x_voc0712.py'
folder="./static"

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


def detect_objects(image_path):
    model = init_detector(config_fname, checkpoint_file)
    result = inference_detector(model, image_path)

    save_dir=os.path.join('./static/images',"result.jpg")
    show_result(image_path, result, model.CLASSES,
            score_thr=0.8, show=False,out_file=save_dir)

    # Comment for result_data{}:
    # + 'bbox' can be used with len(bbox[0]) to judge if failure detected
    # + 'result' point to prediction output imge(result.jpg) path
    result_data = {'bbox': result,
                   'result': save_dir}
    return result_data


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
      img="./static/images/original.jpg"

      result_data = detect_objects(img)
      result= result_data['result']

      timestamp = str(time.time()) # add timestamp to image url to make Browser refresh without using cached image
      result = result + "?" + timestamp # predict output image
      img_timestamp = "./static/images/original.jpg?" + timestamp # uploaded original image

      bbox = result_data['bbox']

    photo_form = PhotoForm(request.form)

    if len(bbox[0]) == 0:
      print('..................green: ', len(bbox[0]))
      return render_template('upload.html',photo_form=photo_form, result=result, color='green', original=img_timestamp)
    else:
      print('..................red: ', len(bbox[0]))
      return render_template('upload.html',photo_form=photo_form, result=result, color='red', original=img_timestamp)



if __name__ == '__main__':
  app.run(host='192.168.117.212', port=5000, debug=True)
