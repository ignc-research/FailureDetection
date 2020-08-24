# Convert training set according to trainval.txt and test.txt
import sys
import os
import json
import xml.etree.ElementTree as ET
import glob
from shutil import copyfile
import shutil
from tqdm import tqdm

START_BOUNDING_BOX_ID = 1
# If necessary, pre-define category and its id
PRE_DEFINE_CATEGORIES = {"good": 1, "medium": 2, "bad": 3}

def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        #return int(filename) # just use filename as image_id
        return filename
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.
    
    Arguments:
        xml_files {list} -- A list of xml file paths.
    
    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_files, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    
    for xml_file in tqdm(xml_files):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)
        image = {
            "file_name": os.path.basename(filename.replace("\\","/")),
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        ## Currently we do not support segmentation.
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
            ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

def get_split_xml_files(xml_files, split_file):
        with open(split_file) as f:
            file_names = [line.strip() for line in f.readlines()]
        return [xml_file for xml_file in xml_files if os.path.splitext(os.path.basename(xml_file))[0] in file_names]

def cleanup_folders():
    '''
    1. delete all related folders
    2. create needed folders
    '''
    paths = [
        "COCO/DIR/ImageSets/Main",
        "COCO/DIR/annotations",
        "COCO/DIR/train2019",
        "COCO/DIR/val2019",
    ]
    for folder in ['COCO']:
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)

    for folder in paths:
        if not os.path.exists(folder):
            os.makedirs(folder)

def split_trainval():

    split_config = {"val": "./COCO/DIR/ImageSets/Main/test.txt",
                    "train": "./COCO/DIR/ImageSets/Main/trainval.txt"}

    for split_file in ["train", "val"]:
        for line in open(split_config[split_file]):
            img_name = line.strip() + ".jpg"
            img_src = "VOC2007_augmented/JPEGImages/" + img_name
            img_target = "COCO/DIR/" + split_file + "2019/" + img_name
            #print(img_src)
            if os.path.isfile(img_src):
                #print("***"+ split_file + "***" + ": Copy " + img_name)
                copyfile(img_src, img_target)
            else:
                print("no")
            #print(img_target)

def do_voc2coco():
    '''
    generate COCO json format annotations from Pascal VOC xml files
    1. instances_train2019.json -> VOC2007_augmented/ImageSets/Main/trainval.txt
    2. instances_val2019.json -> VOC2007_augmented/ImageSets/Main/test.txt
    '''
    xml_dir = "VOC2007_augmented/Annotations/"
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))

    train_split_file = "VOC2007_augmented/ImageSets/Main/trainval.txt"
    val_split_file = "VOC2007_augmented/ImageSets/Main/test.txt"
    copyfile(train_split_file, 'COCO/DIR/ImageSets/Main/trainval.txt')
    copyfile(val_split_file, 'COCO/DIR/ImageSets/Main/test.txt')

    train_xml_files = get_split_xml_files(xml_files, train_split_file)
    val_xml_files = get_split_xml_files(xml_files, val_split_file)

    print("Convert train data annotaions xml listed in trainval.txt to instances_train2019.json......")
    convert(train_xml_files, "COCO/DIR/annotations/instances_train2019.json")
    print("Convert val data annotations xml listed in test.txt instances_val2019.json......")
    convert(val_xml_files, "COCO/DIR/annotations/instances_val2019.json")

if __name__ == "__main__":
    cleanup_folders()
    do_voc2coco()
    split_trainval()

