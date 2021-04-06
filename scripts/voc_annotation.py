import os
import argparse
import xml.etree.ElementTree as ET
import json

def convert_voc_annotation(data_path, data_type, anno_path, class_num, use_difficult_bbox=True):

    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]
    num_images = int(0.05*len(image_inds))

    with open(anno_path, 'a') as f:
        images = image_inds[:num_images] if data_type != 'test' else image_inds
        for image_ind in images:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_name = obj.find('name').text.lower().strip()
                class_ind = classes.index(class_name)
                if not class_name in class_num.keys():
                    class_num[class_name] = 1
                else:
                    class_num[class_name] += 1
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            # print(annotation)
            f.write(annotation + "\n")
    print(len(images))
    return len(image_inds), class_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/chenlei/thesis/tensorflow-yolov3/data/VOC/")
    parser.add_argument("--train_annotation", default="/home/chenlei/thesis/tensorflow-yolov3/data/dataset/voc_train.txt")
    parser.add_argument("--test_annotation",  default="/home/chenlei/thesis/tensorflow-yolov3/data/dataset/voc_test.txt")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation):os.remove(flags.test_annotation)

    class_num = {}
    num1, class_num = convert_voc_annotation(os.path.join(flags.data_path, 'train/VOCdevkit/VOC2007'), 'trainval', flags.train_annotation, class_num, False)
    num2, class_num = convert_voc_annotation(os.path.join(flags.data_path, 'train/VOCdevkit/VOC2012'), 'trainval', flags.train_annotation, class_num, False)
    test_class_num = {}
    num3, test_class_num = convert_voc_annotation(os.path.join(flags.data_path, 'test/VOCdevkit/VOC2007'),  'test', flags.test_annotation, test_class_num, False)
    print('=> The number of image for train is: %d\tThe number of image for test is:%d' %(num1+num2, num3))
    json.dump(class_num, open('/home/chenlei/thesis/tensorflow-yolov3/data/dataset/class_num_train.json', 'w'))
    json.dump(test_class_num, open('/home/chenlei/thesis/tensorflow-yolov3/data/dataset/class_num_test.json', 'w'))



