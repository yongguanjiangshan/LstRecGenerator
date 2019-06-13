# -*- coding: UTF-8 -*-

import os
import sys
from xml.dom import minidom  #处理xml数据
from os.path import join
import argparse
import numpy as np

args = argparse.ArgumentParser()
args.add_argument("-d", "--dir",
                 help="path to target dataset.")
args.add_argument("-l", "--lst",
                 help="name of .lst file to be written.")
args.add_argument("-c", "--classFile",
                 help="path to txt file given classNames.")
args = args.parse_args()

#首先定义一个读取xml文件的函数：
def xmlDecode(path, classNames):
    annotation = minidom.parse(path)

    size = annotation.getElementsByTagName('size')[0]
    width = size.getElementsByTagName('width')[0].firstChild.data
    height = size.getElementsByTagName('height')[0].firstChild.data

    obj = annotation.getElementsByTagName('object')[0]
    cla = obj.getElementsByTagName('name')[0].firstChild.data  #类别
    bndbox = obj.getElementsByTagName('bndbox')[0]              #坐标
    x1 = bndbox.getElementsByTagName('xmin')[0].firstChild.data
    x2 = bndbox.getElementsByTagName('xmax')[0].firstChild.data
    y1 = bndbox.getElementsByTagName('ymin')[0].firstChild.data
    y2 = bndbox.getElementsByTagName('ymax')[0].firstChild.data
    

    width = int(width)
    height = int(height)
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    result = [classNames.index(cla), (width,height), (x1,y1), (x2,y2)]
    return result

def write_line(img_path, im_shape, boxes, ids, idx):
    h, w = im_shape
    # for header, we use minimal length 2, plus width and height
    # with A: 4, B: 5, C: width, D: height
    A = 4
    B = 5
    C = w
    D = h
    # concat id and bboxes
    labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
    # normalized bboxes (recommanded)
    labels[:, (1, 3)] /= float(w)
    labels[:, (2, 4)] /= float(h)
    # flatten
    labels = labels.flatten().tolist()
    str_idx = [str(idx)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_labels = [str(x) for x in labels]
    str_path = [img_path]
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    return line

def _parse_voc_anno(filename, classNames):
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    height = int(tree.find('size').find('height').text)
    width = int(tree.find('size').find('width').text)
    objects = []
    all_boxes = []
    all_ids = []
    for obj in tree.findall('object'):
        obj_dict = dict()
        obj_dict['name'] = obj.find('name').text
        obj_dict['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        box= [int(float(bbox.find('xmin').text)),
                            int(float(bbox.find('ymin').text)),
                            int(float(bbox.find('xmax').text)),
                            int(float(bbox.find('ymax').text))]
        obj_dict['bbox'] = box
        objects.append(obj_dict)
        all_boxes.append(box)
        cls_name = obj.find('name').text
        cls_id = classNames.index(obj.find('name').text)

        all_ids.append(cls_id)
    return height, width, objects, all_boxes, all_ids

def getClassNames(txtFile):
    with open(txtFile, "r") as f:
        classNames = f.readlines()[0].split(" ")
    return list(filter(None, classNames))

if __name__ == "__main__":
    #定义保存数据和标签文件夹路径
    path = args.dir
    lst_file_name = args.lst
    #获取xml文件中出现的所有类别名称
    classNames = getClassNames(args.classFile)
    #假设图片名和对应的标签名称一致，这里直接替换xml为jpg
    #format:0  4  5  640(width)  480(height)  1(class)  0.1  0.2  0.8  0.9(xmin, ymin, xmax, ymax)  2  0.5  0.3  0.6  0.8  data/xxx.jpg
    names = os.listdir(path)
    lst = []
    i=0
    with open(lst_file_name,'w') as fw:
        for name in names:
            if name.endswith('.xml'):
                h,w,objs,all_boxes,all_ids = _parse_voc_anno(join(path, name), classNames)
                img_name = join(path, name.replace('xml','jpg'))
                shape = h,w
                all_boxes_np = np.array(all_boxes)
                all_ids_np = np.array(all_ids)
                line = write_line(img_name, shape, all_boxes_np, all_ids_np, i)
                print(line)
                fw.write(line)
                i+=1
    fw.close()
