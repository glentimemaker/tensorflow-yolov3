import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOV3
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
class YoloTest(object):
    def __init__(self):
        self.input_size       = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold  = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path  = cfg.PSEUDO.ANNO_PATH
        self.weight_file      = cfg.TEST.WEIGHT_FILE
        self.write_image      = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label       = cfg.TEST.SHOW_LABEL

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')

        model = YOLOV3(self.input_data, self.trainable)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input_data: image_data,
                self.trainable: False
            }
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes

    def evaluate(self):
        pseudo_data = cfg.PSEUDO.TEMP_DATA
        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        os.mkdir(self.write_image_path)
        lines = []
        # counter = 0
        with open(self.annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                # counter += 1
                # if counter > 100:
                #     break
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)

                bboxes_pr = self.predict(image)

                all_bbox = []
                for bbox in bboxes_pr:
                    coor = ','.join(str(int(x)) for x in bbox[:4])
                    score = bbox[4]
                    class_ind = str(int(bbox[5]))
                    if score > cfg.PSEUDO.THRESHOLD:
                        print('=> predict result of %s:' % image_name)
                        if not all_bbox:
                            all_bbox.append(image_path)
                        coor_class = coor + ',' + class_ind
                        all_bbox.append(coor_class)
                        if self.write_image:
                            image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                            cv2.imwrite(self.write_image_path + image_name, image)
                            print(self.write_image_path + image_name)
                        lines.append(' '.join(all_bbox))
            with open(pseudo_data, 'w') as f:
                for line in lines:
                    f.write(line)
                    f.write('\n')

if __name__ == '__main__':
    yolotest = YoloTest()
    yolotest.evaluate()

    import shutil
    with open(cfg.PSEUDO.TRAIN_DATA, 'wb') as wfd:
        for f in [cfg.TRAIN.ANNOT_PATH, cfg.PSEUDO.TEMP_DATA]:
            with open(f, 'rb') as fd:
                shutil.copyfileobj(fd, wfd)





