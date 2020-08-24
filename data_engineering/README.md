# Data Augmentation Scripts

Find dataset in Server 46: /home/xueliang/dataset/BAM_new_added_24.08

<img src="../../../../Desktop/Screenshot%202020-08-24%20at%2013.49.01.png" alt="Screenshot 2020-08-24 at 13.49.01" style="zoom:40%;" />

## Original Images

Extracted with Matlab from raw '.mat' files. Refer to the 'matlab_scripts' folder.



## Labeling

1. Install LabelImg from Github.
2. Images quality referenced in file 'Schweißung-gut_mittel_schlecht.pdf'.  ie, 'good', 'medium', 'bad'. 
3. Specifically, all images from same one film have same quality label.
4. Each image has only one target/label.

```
- [ ] add labeling img
```



## Augmentation

Put labeled data inside folder 'VOC2007':

1. Inside folder VOC2007, 'JPEGImages' for all images, 'Annotations' for all annotation files.
2. File 'Schweißung-gut_mittel_schlecht.pdf' references to images quality used for labeling,
3. Script '1_voc_augmentation.py' for Pascal VOC data augmentation.
4. Script '2_voc2coco.py' converts augmented VOC data to COCO.



## Potential problems

Hiden files like '.DS_*' could cause the script crashing.