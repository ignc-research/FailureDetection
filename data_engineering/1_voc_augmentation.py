import os
from os import remove
import os.path
from tqdm import tqdm
import random
import shutil

import xml.etree.ElementTree as ET

import numpy as np
import cv2
from PIL import Image

data_path = "VOC2007/JPEGImages/"
xml_path = "VOC2007/Annotations"
img_path = "VOC2007_augmented/JPEGImages/"
out_path = "tmp/bb_box.txt"

bb_box_path = "tmp/bb_box.txt"
single_augmented_txt = "tmp/splited_txt/"
file_name_path = "tmp/file_name.txt"

ANNOTATIONS_DIR_PREFIX = "tmp/splited_txt/"
DESTINATION_DIR = "VOC2007_augmented/"

xmls = os.listdir(xml_path)
if '.DS_Store' in xmls:
    xmls.remove('.DS_Store')
xml_paths = [os.path.join(xml_path, s) for s in xmls]

def cleanup_folders():
    '''
    1. delete all related folders
    2. create needed folders
    '''
    paths = [
        "VOC2007_augmented/Annotations",
        "VOC2007_augmented/ImageSets/Main",
        "VOC2007_augmented/JPEGImages",
        "tmp/splited_txt"
    ]
    for folder in ['VOC2007_augmented', 'tmp']:
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)

    for folder in paths:
        if not os.path.exists(folder):
            os.makedirs(folder)

def pca_color_augmentation(image):
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8
    img = image.reshape(-1, 3).astype(np.float32)
    sf = np.sqrt(3.0/np.sum(np.var(img, axis=0)))
    img = (img - np.mean(img, axis=0))*sf
    cov = np.cov(img, rowvar=False)  # calculate the covariance matrix
    # calculation of eigen vector and eigen value
    value, p = np.linalg.eig(cov)
    rand = np.random.randn(3)*0.08
    delta = np.dot(p, rand*value)
    delta = (delta*255.0).astype(np.int32)[np.newaxis, np.newaxis, :]
    img_out = np.clip(image+delta, 0, 255).astype(np.uint8)
    return img_out

def img_augmentation():
    pwd_lines = []
    element_objs = []

    for xml_file in tqdm(xml_paths):
        #print(xml_file)
        et = ET.parse(xml_file)
        element = et.getroot()
        element_objs = element.findall('object')
        element_filename = element.find('filename').text
        base_filename = os.path.join(data_path, element_filename)
        #print(base_filename)
        img = cv2.imread(base_filename)

        img_split = element_filename.strip().split('.jpg')

        for element_obj in element_objs:
            # return name tag ie class of disease from xml file
            class_name = element_obj.find('name').text

            obj_bbox = element_obj.find('bndbox')
            # print(obj_bbox)
            x1 = int(round(float(obj_bbox.find('xmin').text)))
            y1 = int(round(float(obj_bbox.find('ymin').text)))
            x2 = int(round(float(obj_bbox.find('xmax').text)))
            y2 = int(round(float(obj_bbox.find('ymax').text)))

            # if you specify range(1) total number of augmented data is 6 for one image
            # 6 types of augmentation means pca, horizontal and vertical flip, 3 rotation
            # if you specify range(2) total number of augmented data is 12 for one image
            for color in range(1):
                img_color = pca_color_augmentation(img)
                color_name = img_split[0] + '-color' + str(color)
                color_jpg = color_name + '.jpg'
                # join with augmented image path
                new_path = os.path.join(img_path, color_jpg)
                lines = [color_jpg, ',', str(x1), ',', str(y1), ',', str(
                    x2), ',', str(y2), ',', class_name, '\n']
                pwd_lines.append(lines)
                if not os.path.isfile(new_path):
                    cv2.imwrite(new_path, img_color)

                # for horizontal and vertical flip
                f_points = [0, 1]
                for f in f_points:
                    f_img = cv2.flip(img_color, f)
                    h, w = img_color.shape[:2]

                    if f == 1:
                        f_x1 = w-x2
                        f_y1 = y1
                        f_x2 = w-x1
                        f_y2 = y2
                        f_str = 'f1'
                    elif f == 0:
                        f_x1 = x1
                        f_y1 = h-y2
                        f_x2 = x2
                        f_y2 = h-y1
                        f_str = 'f0'

                    new_name = color_name + '-' + f_str + '.jpg'
                    new_path = os.path.join(img_path, new_name)

                    lines = [new_name, ',', str(f_x1), ',', str(f_y1), ',', str(
                        f_x2), ',', str(f_y2), ',', class_name, '\n']
                    pwd_lines.append(lines)
                    if not os.path.isfile(new_path):
                        cv2.imwrite(new_path, f_img)

                # for angle 90
                img_transpose = np.transpose(img_color, (1, 0, 2))
                img_90 = cv2.flip(img_transpose, 1)
                h, w = img_color.shape[:2]
                angle_x1 = h - y2
                angle_y1 = x1
                angle_x2 = h - y1
                angle_y2 = x2
                new_name = color_name + '-' + 'rotate_90' + '.jpg'
                new_path = os.path.join(img_path, new_name)
                lines = [new_name, ',', str(angle_x1), ',', str(angle_y1), ',', str(
                    angle_x2), ',', str(angle_y2), ',', class_name, '\n']
                pwd_lines.append(lines)
                if not os.path.isfile(new_path):
                    cv2.imwrite(new_path, img_90)

                # for angle 180
                img_180 = cv2.flip(img_color, -1)
                ang_x1 = w - x2
                ang_y1 = h - y2
                ang_x2 = w - x1
                ang_y2 = h - y1
                new_name_180 = color_name + '-' + 'rotate_180' + '.jpg'
                new_path_180 = os.path.join(img_path, new_name_180)
                lines_180 = [new_name_180, ',', str(ang_x1), ',', str(
                    ang_y1), ',', str(ang_x2), ',', str(ang_y2), ',', class_name, '\n']
                pwd_lines.append(lines_180)
                if not os.path.isfile(new_path_180):
                    cv2.imwrite(new_path_180, img_180)

                # for angle 270
                img_transpose_270 = np.transpose(img_color, (1, 0, 2))
                img_270 = cv2.flip(img_transpose_270, 0)
                an_x1 = y1
                an_y1 = w - x2
                an_x2 = y2
                an_y2 = w - x1
                new_name_270 = color_name + '-' + 'rotate_270' + '.jpg'
                new_path_270 = os.path.join(img_path, new_name_270)
                lines_270 = [new_name_270, ',', str(an_x1), ',', str(
                    an_y1), ',', str(an_x2), ',', str(an_y2), ',', class_name, '\n']
                pwd_lines.append(lines_270)
                if not os.path.isfile(new_path_270):
                    cv2.imwrite(new_path_270, img_270)

    # print(pwd_lines)
    if len(pwd_lines) > 0:
        with open(out_path, 'w') as f:
            for line in pwd_lines:
                f.writelines(line)

    print('Image Augmentation ... Done!')

def make_dataset(input_path):
    all_imgs = {}
    #print("the input path is", input_path)
    with open(input_path,'r') as f:
        #print('Parsing annotation files')
        for line in f:
            line_split = line.strip().split(',')
            (filename,x1,y1,x2,y2,class_name) = line_split
            if filename not in all_imgs:
                all_imgs[filename] = {}
                
                all_imgs[filename]['file_name_path'] = filename
                all_imgs[filename]['bboxes'] = []
                
            all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

        all_data = []

        for key in all_imgs:
            all_data.append(all_imgs[key])
            
        return all_data

def read_txtfile():
    dataset = make_dataset(bb_box_path)

    for i in range(len(dataset)):
        dataset_path = dataset[i]['file_name_path']
        dataset_name = dataset_path.split(',')[0]
        sep_dataset = dataset[i]['bboxes']
        
        with open(file_name_path, 'a') as fn:
            file_name = "{}\n".format(dataset_name)
            fn.write(file_name)
            
        with open(single_augmented_txt + str(dataset_name)+ ".txt", 'w') as f:
            for j in range(len(sep_dataset)):
                # save_dataset = sep_dataset[j]
                data = "{} {} {} {} {} \n".format(sep_dataset[j]['class'],sep_dataset[j]['x1'],sep_dataset[j]['y1'], sep_dataset[j]['x2'], sep_dataset[j]['y2'])
                f.write(data)

def create_root(file_prefix, width, height):
    root = ET.Element("annotations")
    ET.SubElement(root, "filename").text = "{}.jpg".format(file_prefix)
    ET.SubElement(root, "folder").text = "images"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    return root

def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])
    return root

def create_file(file_prefix, width, height, voc_labels):
    root = create_root(file_prefix, width, height)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root)
    tree.write(DESTINATION_DIR + "Annotations/{}.xml".format(file_prefix))

def read_file(file_path):
    file_prefix = file_path.split(".jpg")[0]
    file_path_data = ANNOTATIONS_DIR_PREFIX + file_path
    #print("the file path data", file_path_data)
    image_file_name = "{}.jpg".format(file_prefix)
    img = Image.open("{}/{}".format(DESTINATION_DIR + 'JPEGImages', image_file_name))
    w, h = img.size
    with open(file_path_data, 'r') as file:
        lines = file.readlines()
        voc_labels = []
        for line in lines:
            voc = []
            line = line.strip()
            data = line.split()
            #print(data[0],data[1],data[2],data[3],data[4])
            voc.append(data[0])
            voc.append(data[1])
            voc.append(data[2])
            voc.append(data[3])
            voc.append(data[4])
            voc_labels.append(voc)
            #print(voc_labels)
        create_file(file_prefix, w, h, voc_labels)
    #print("Processing complete for file: {}".format(file_path))

def anno_augmentation():
    
    read_txtfile()

    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)
    for filename in tqdm(os.listdir(ANNOTATIONS_DIR_PREFIX)):
        #print(filename)
        if filename.endswith('txt'):
            read_file(filename)
        else:
            print("Skipping file: {}".format(filename))
    
    if os.path.exists('tmp'):
        shutil.rmtree('tmp', ignore_errors=True)
    print("Annotations Augmentation ... Done!")


xmlfilepath = "VOC2007_augmented/Annotations"
imageSetsPath = "VOC2007_augmented/ImageSets/Main"
trainval_percent=0.8
train_percent=0.8
def split_trainval():

    total_xml = os.listdir(xmlfilepath)
    if '.DS_Store' in total_xml:
        total_xml.remove('.DS_Store')
    num=len(total_xml)
    list=range(num)
    #
    tv=int(num*trainval_percent)
    tr=int(tv*train_percent)
    trainval= random.sample(list,tv)
    train=random.sample(trainval,tr)
    
    print("trainval size:", tv)
    print("test size:", num - tv)
    ftrainval = open(os.path.join(imageSetsPath,'trainval.txt'), 'w')
    ftest = open(os.path.join(imageSetsPath,'test.txt'), 'w')
    ftrain = open(os.path.join(imageSetsPath,'train.txt'), 'w')
    fval = open(os.path.join(imageSetsPath,'val.txt'), 'w')

    for i  in list:
        name=total_xml[i][:-4]+'\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest .close()


if __name__ == "__main__":
    cleanup_folders()
    print("Start image file augmentation:......")
    img_augmentation()
    print("Start annotation file augmentation:......")
    anno_augmentation()
    split_trainval()
