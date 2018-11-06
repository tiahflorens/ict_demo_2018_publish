import tensorflow as tf
from lib.utils.simple_op import draw_trajectory, velocity
from lib.nets import ssd_vgg_300
from lib.preprocessing import ssd_vgg_preprocessing
import numpy as np
import cv2
from lib.nets import np_methods

slim = tf.contrib.slim
from lib.utils.var import ckpt_filename
import random

stamp_list = [[], [], [], []]

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.2)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NCHW'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


def tracking_module(mot_tracker, track_list, bbox_list, frame):
    if len(track_list) > 50:
        # TODO check, delete 30 tracks if list is longer than 50.
        # however, if tracker uses every tracks in the list, the # tracks used in tracking is not consistant.
        del track_list[0:30]

    if len(bbox_list) != 0:
        trackers = mot_tracker.update(bbox_list)
        track_list.append(trackers)
        for d in trackers:
            d = d.astype(np.int32)
            if len(track_list) > 5:
                track_list, trajectories = draw_trajectory(track_list)

                for _ttmp in trajectories:
                    cv2.polylines(frame, np.array(_ttmp[1][0], np.int32).reshape((-1, 1, 2)), True,
                                  (255, 0, 0),
                                  lineType=cv2.LINE_AA, thickness=4)
                    _vel = abs(velocity(np.array(_ttmp[1][0], np.int32).reshape((-1, 1, 2))))

                    if _vel < 1.0:
                        msg2 = 'Standing'
                    elif _vel < 20.0:
                        msg2 = 'Walking'
                    else:
                        msg2 = 'Running'
                    msg_size, baseLine1 = cv2.getTextSize(msg2, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
                    # print(msg_size)
                    cv2.rectangle(frame, (d[2], d[1] + 17), (d[2] + msg_size[0], d[1] + 17 + msg_size[1]), (255, 0, 0),
                                  -1)
                    cv2.putText(frame, msg2, (d[2], d[1] + 16 + msg_size[1]), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                (255, 255, 255), 1)
                    # cv2.putText(frame, msg2, (int(d[0]), int(d[1]) - 15), cv2.FONT_HERSHEY_DUPLEX, 0.7,(255, 0, 0), 1)


def process_image(img, select_threshold=0.25, nms_threshold=.25, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    # TODO same paramss usesd twice
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


def process_ssd_result(classes, bboxes, img_shape):
    thing_list, person_list = [], []

    for cls, bbox in zip(classes, bboxes):
        p1 = (int(bbox[0] * img_shape[0]), int(bbox[1] * img_shape[1]))
        p2 = (int(bbox[2] * img_shape[0]), int(bbox[3] * img_shape[1]))

        if cls == 15:  # class for  person
            person_list.append(list(np.hstack((np.hstack((p1[::-1], p2[::-1])), random.uniform(0.97, 0.999)))))
        else:
            thing_list.append(list(np.hstack((np.hstack((p1[::-1], p2[::-1])), random.uniform(0.97, 0.999)))))
    return person_list, thing_list