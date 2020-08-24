import sys
import tempfile
import os

#from decorator import requires_auth

import numpy as np
from PIL import Image
import cv2
import mmcv
from mmcv.runner import load_checkpoint
import mmcv.visualization.image as mmcv_image
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, init_detector

import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"

#checkpoint_file = './models/epoch_14.pth'
checkpoint_file = '/home/xueliang/mmdetection/data/work_dirs/faster_rcnn_bam_old/latest.pth'
config_fname = '/home/xueliang/mmdetection/data/configs/faster_rcnn_r50_fpn_1x_voc0712.py'



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


if __name__ == '__main__':
  app.run(host='192.168.117.46', port=5000, debug=True)
