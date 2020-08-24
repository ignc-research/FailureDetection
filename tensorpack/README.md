# How to train an object detection model with Tensorpack on own dataset

Here should be the installation folder of Tensorpack. Tensorpack is a training interface based on TensorFlow.

- [Official tutorial](https://tensorpack.readthedocs.io/tutorial/index.html#user-tutorials).

## How to use

1. Prepare COCO dataset(depends on the built-in configuration, could be changed to VOC)

2. Adapt training parameters(class names) & Hyper-parameters(training scale).

   - Class names: 'tensorpack/examples/FasterRCNN/dataset/coco.py'
   - Hyper-parameters: 'tensorpack/examples/FasterRCNN/config.py'

3. Start training, refer to official demo(better with shell file):

   ```shell
   # start normal(like Faster/Cascade/Retina) training
   # change all data & results related paths to 'mydata'
   ./train.py --config \
       BACKBONE.WEIGHTS=/path/to/ImageNet-R50-AlignPadding.npz \
       DATA.BASEDIR=/path/to/COCO/DIR \
       [OTHER-ARCHITECTURE-SETTINGS]
   # more parameters can be adapted to own configurations
   ```

   

4. Monitor training: Tensorboard from Tensorflow.

   ```shell
   # Start remote tensorboard service with browser:
   # train_log for trained model path
   (ssh)Remote Server: tensorboard --logdir train_log/03122019_2 --host 192.168.117.46 --port 6006 
   
   #In any case need to kill tensorboard with it’s process name:
   pkill -f my_pattern
   ```

5. Reference:

   ```shell
   ./predict.py --predict input1.jpg input2.jpg --load /path/to/Trained-Model-Checkpoint --config SAME-AS-TRAINING
   ```

   The above codes test only one image. The 'batch_predict.py' is written for detecting all images inside test folder,  initing predictor and doing prediction in a loop. Four parameters need to be specified, i.g:

   - Model path: session_init=SmartInit("mydata/train_log_bam/20.01.2020/checkpoint")
   - checkpoint file: it’s a txt, in first line specify model number usded for test
   - test img path: source_folder = "data/forTest_bam/"
   - result output path: result_folder = source_folder + "../" + "result/"

6. Deployment, refer to 'Deployment' folder.

7. Evaluation on COCO:

   ```shell
   ./predict.py --evaluate output.json --load /path/to/Trained-Model-Checkpoint \
       --config SAME-AS-TRAINING
   ```

   

### TODO

```
- [ ] Tensorpack && MMDetection version problem, Server is setup with tp v0.9x and mmd v1.0x, both are updated.
- [ ] more adaptation details
- [ ] ...
```

## Work environment

Hardware:

1. LAN cabel on office desk, for Internet and GPU server

2. Lab GPU server, accessiable with lab local network. Connect to server: 

   ```shell
   $ ssh username@192.168.117.*
   prompt for password
   ```

3. Google Colab(free GPU from google)

Software:

1. ssh: connect to remote server with ip and user account
2. tmux: keep shell session running on server when local PC shutdown
3. $ scp: copy file between local and server(FileZilla recommended)
4. Anaconda(manage Pytorch and Tensorflow environments)
5. Jupyter Notebook or Jupyterlab(recommended)
6. VS code or Pycharm ssh remote development



# Demo: Train FasterRCNN with tensorpack

Official [guide](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN). Our tested main dependencies(available on Server 46.):

- Ubuntu 16.04 LTS
- CUDA 10.0
- cuDNN 7.6.1
- opencv 3.4.2
- numpy 1.16.5
- tensorflow-gpu 1.13.2 (recommded)
- tensorpack v0.9.8
- Anaconda/python 3.6.9

Tensorpack has many network implementations, here we use FasterRCNN for our case. The relevent scripts are located in tensorpack/examples/FasterRCNN. To start training for our own dataset, we need to modify some parameters, either with command parameters or within the following 3 scripts:

- in "tensorpack/examples/**FasterRCNN/config.py**", to define the name of our train and valida- tion directory, change: 

  C.DATA.TRAIN = (’coco train2017’,) to C.DATA.TRAIN = (’train2019’,)

  C.DATA.VAL = (’coco val2017’,) to C.DATA.VAL = (’val2019’,)

- tensorpack/examples/FasterRCNN/**dataset/COCO.py**, we have only one class "failure", change the register_coco function like below:

  ```python
  def register_coco(basedir):
      class_names = ["BG", "failure"] # "BG", background is needed
      for split in ["train2019", "val2019"]:
          DatasetRegistry.register(split, lambda x=split: COCODetection(basedir, x))
          DatasetRegistry.register_metadata(split, 'class_names', class_names)
  ```

After put prepared COCO dataset and pre-trained [ImageNet ResNet model](http://models.tensorpack.com/#FasterRCNN) "ImageNet-R50-AlignPadding.npz" from tensorpack model zoo in tensorpack/examples/FasterRCNN/data/, we can start training with commands:

```shell
#!/bin/sh
python train.py \
    --logdir data/train_log/ \
    --config \
        DATA.BASEDIR=data/COCO/DIR \
        BACKBONE.WEIGHTS=data/ImageNet-R50-AlignPadding.npz \
        MODE_MASK=False \
        TRAIN.CHECKPOINT_PERIOD=1
```

The meaning of each parameter can be found in "tensorpack/examples/FasterRCNN/**train.py**" script, and the tranned model is all in --logdir specified folder. the trainning logs are standard tensorflow output, the details and meaning of these files can be found in tensorflow user manual. When the model is produced, when can test with command:

```shell
#!/bin/sh
python predict.py \
		--predict test.jpg \
		--load /path/to/Trained-Model-Checkpoint \
		--config \
				SAME-AS-TRAINING
```

The prediction output path is specified in **tensorpack/examples/FasterRCNN/predict.py** script, and it’ll be put in current location by default with name "output.png".



### TO Change bbox color used in prediction:

related function:

```python
result = predict_image(img, pre_func)
```

