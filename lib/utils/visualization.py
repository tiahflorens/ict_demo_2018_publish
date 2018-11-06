# Copyright 2017 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import copy
import random

import cv2
import matplotlib.cm as mpcm
import numpy as np
from lib.utils.var import *

SZ = 20
bin_n = 64  # Number of bins


# =========================================================================== #
# Some colormaps.
# =========================================================================== #
def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i * dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors


# colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (255, 0, 0), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

clssses_string = ['Aeroplanes', 'Bicycles', 'Birds', 'Boats', 'Bottles', 'Buses',
                  'Cars', 'Cats', 'Chairs', 'Cows', 'Dining tables', 'Dogs', 'Horses',
                  'Motorbikes', 'People', 'Potted plants', 'Sheep', 'Sofas', 'Trains',
                  'TV/Monitors']

list_for_detect = [2, 7, 15]
# list_for_detect = [ 15]


def int2round(src):
    """
    returns rounded integer recursively
    :param src:
    :return:
    """
    if isinstance(src, float):
        return int(round(src))

    elif isinstance(src, tuple):
        res = []
        for i in range(len(src)):
            res.append(int(round(src[i])))
        return tuple(res)

    elif isinstance(src, list):
        res = []
        for i in range(len(src)):
            res.append(int2round(src[i]))
        return res
    elif isinstance(src, int):
        return src
    if isinstance(src, str):
        return int(src)


def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist


def predict_fight(mlp_model, _crop_img):
    hog_hist = np.float32(np.reshape(hog(_crop_img), (1, -1)))
    result = mlp_model.predict(hog_hist)
    prob_result = mlp_model.predict_proba(hog_hist)
    return result, prob_result


def bboxes_draw_on_img_for_eventv2(img, classes, scores, bboxes, mlp_model, colors=colors_tableau, thickness=4):
    shape = img.shape
    _circle_list = []
    _pedestrian_list = []
    org_img = np.copy(img)
    _bbox_list = []
    _unified_list = []
    for i in range(bboxes.shape[0]): # the number of bbox prediction.
        if classes[i] in list_for_detect: # class is in bicycles, person, car

            bbox = bboxes[i]
            color = colors[classes[i]]

            # Draw bounding box...
            p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            # p12 = p1[::-1]
            # print(p1, p2, p12)
            # cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
            _center = tuple((np.array(p1[::-1]) + np.array(p2[::-1])) / 2) # p1[::-1] is reversing (x,y) -> (y,x)
            _circle_list.append(_center)
            # crop_img = cv2.resize(org_img[p1[0]:p2[0], p1[1]:p2[1]], (240, 640), interpolation=cv2.INTER_CUBIC)
            # predict, likelihood = predict_fight(mlp_model, crop_img)


            if classes[i] == 15: # center x,y of person
                # _circle_list.append(_center)
                _bbox_list.append(list(np.hstack((np.hstack((p1[::-1], p2[::-1])), random.uniform(0.97, 0.999)))))
                _pedestrian_list.append([p1[::-1], p2[::-1], _center])

            # cv2.circle(img, int2round(_center), 3, color, -1)
            # if likelihood[0, 1] > likelihood[0, 0]:
            #     s = '%s/%.3f-Abn' % (clssses_string[int(classes[i]) - 1], scores[i])
            # else:
            #     s = '%s/%.3f-Nor' % (clssses_string[int(classes[i]) - 1], scores[i])

            # s = '%s/%.3f' % (clssses_string[int(classes[i]) - 1] ,scores[i])
            s_class = (clssses_string[int(classes[i]) - 1])
            s_class_size,baseLine1 = cv2.getTextSize(s_class, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)

            up_right = (p2[::-1][0], p1[::-1][1])
            # print(s_class_size)  # w, 16
            # cv2.rectangle(img, up_right, (up_right[0] + s_class_size[0], up_right[1]+ s_class_size[1]),color, -1)
            # cv2.putText(img, s_class, (up_right[0], up_right[1]+s_class_size[1]), cv2.FONT_HERSHEY_DUPLEX, 0.7 ,(255, 255, 255), 1)

            # sec_up_right = (up_right[0], up_right[1]+s_class_size[1])
            # cv2.rectangle(img, sec_up_right, (sec_up_right[0] + s_score_size[0], sec_up_right[1] + s_score_size[1]),color, -1)
            # cv2.putText(img, s_score, (up_right[0], up_right[1]+s_score_size[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

            # p1 = (p1[0] - 5, p1[1])d
            # txtSize, baseLine = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
            # cv2.rectangle(img, (p2[::-1][0],p1[::-1][1]), (p2[::-1][0]+txtSize[0], p1[::-1][1]+txtSize[1]),color,-1)


    center_list = copy.copy(_circle_list)


    for _tmp in _circle_list:
        _circle_list.remove(_tmp)
        for _ttmp in _circle_list:
            _t = np.array(_tmp[::-1]) - np.array(_ttmp[::-1])
            _length = np.sqrt(np.square(_t[0]) + np.square(_t[1]))
            if _length < 100.0:
                _line_center = tuple((np.array(_tmp[::]) + np.array(_ttmp[::])) / 2)
                cv2.line(img, int2round(_tmp), int2round(_ttmp), (255, 0, 0), 2)
                cv2.putText(img, '%.3f' % (_length), int2round(_line_center), FONT_FACE, 0.4, RED, 1)
            elif _length < 150.0:
                _line_center = tuple((np.array(_tmp[::]) + np.array(_ttmp[::])) / 2)
                cv2.line(img, int2round(_tmp), int2round(_ttmp), (255, 140, 0), 2)
                cv2.putText(img, '%.3f' % (_length), int2round(_line_center), FONT_FACE, 0.4, ORANGE, 1)
            elif _length < 200.0:
                _line_center = tuple((np.array(_tmp[::]) + np.array(_ttmp[::])) / 2)
                cv2.line(img, int2round(_tmp), int2round(_ttmp), (255, 255, 0), 2)
                cv2.putText(img, '%.3f' % (_length), int2round(_line_center), FONT_FACE, 0.4, YELLOW, 1)
            elif _length < 250.0:
                _line_center = tuple((np.array(_tmp[::]) + np.array(_ttmp[::])) / 2)
                cv2.line(img, int2round(_tmp), int2round(_ttmp), (9, 249, 17), 2)
                cv2.putText(img, '%.3f' % (_length), int2round(_line_center), FONT_FACE, 0.4, LIGHT_GREEN, 1)
    if len(center_list) > 3:
        _spr_or_grp = _ccrowd_estimation(center_list)
    else:
        _spr_or_grp = -1
    return img, np.array(_bbox_list), _spr_or_grp





def _ccrowd_estimation(_center_list):
    _coc = np.mean(np.array(_center_list))
    _dist_avg = 0.0
    _count = 0
    for _tmp in _center_list:
        dist = np.linalg.norm(_coc - _tmp)
        if dist < 300.0: # exclude outlier person, too FAR
            _dist_avg += dist
            _count += 1
    if _count == 0:
        _dist_avg = -1
    else:
        _dist_avg = _dist_avg / _count
    return _dist_avg


def bboxes_draw_on_img_for_eventv2_tune(PARAM_CROWD, img, classes, scores, bboxes, mlp_model, colors=colors_tableau,
                                        thickness=2):
    shape = img.shape
    _circle_list = []
    _pedestrian_list = []
    org_img = np.copy(img)
    _bbox_list = []
    _unified_list = []
    for i in range(bboxes.shape[0]): # the number of bbox prediction.
        if classes[i] in list_for_detect: # class is in bicycles, person, car
            bbox = bboxes[i]
            color = colors[classes[i]]
            # Draw bounding box...
            p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
            _center = tuple((np.array(p1[::-1]) + np.array(p2[::-1])) / 2) # p1[::-1] is reversing (x,y) -> (y,x)
            # _circle_list.append(_center)
            crop_img = cv2.resize(org_img[p1[0]:p2[0], p1[1]:p2[1]], (240, 640), interpolation=cv2.INTER_CUBIC)
            predict, likelihood = predict_fight(mlp_model, crop_img)


            if classes[i] == 15: # person
                _circle_list.append(_center) # center x,y of person
                _bbox_list.append(list(np.hstack((np.hstack((p1[::-1], p2[::-1])), random.uniform(0.97, 0.999)))))
                _pedestrian_list.append([p1[::-1], p2[::-1], _center])
                # _unified_list = [_center, np.hstack((p1[::-1], p2[::-1]))]
                # _unified_list = [_center,np.hstack((p1[::-1],p2[::-1])]
            cv2.circle(img, int2round(_center), 3, color, -1)
            if likelihood[0, 1] > likelihood[0, 0]:
                s = '%s/%.3f-Abn' % (clssses_string[int(classes[i]) - 1], scores[i])
            else:
                s = '%s/%.3f-Nor' % (clssses_string[int(classes[i]) - 1], scores[i])
            p1 = (p1[0] - 5, p1[1])
            cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)



    center_list = copy.copy(_circle_list)
    for _tmp in _circle_list:
        _circle_list.remove(_tmp)
        for _ttmp in _circle_list:
            _t = np.array(_tmp[::-1]) - np.array(_ttmp[::-1])
            _length = np.sqrt(np.square(_t[0]) + np.square(_t[1])) # length between previous and current center.
            if _length < 100.0:
                _line_center = tuple((np.array(_tmp[::]) + np.array(_ttmp[::])) / 2)
                cv2.line(img, int2round(_tmp), int2round(_ttmp), (255, 0, 0), 2)
                cv2.putText(img, '%.3f' % (_length), int2round(_line_center), FONT_FACE, 0.4, (255, 0, 0), 1)
            elif _length < 150.0:
                _line_center = tuple((np.array(_tmp[::]) + np.array(_ttmp[::])) / 2)
                cv2.line(img, int2round(_tmp), int2round(_ttmp), (255, 140, 0), 2)
                cv2.putText(img, '%.3f' % (_length), int2round(_line_center), FONT_FACE, 0.4, (255, 140, 0), 1)
            elif _length < 200.0:
                _line_center = tuple((np.array(_tmp[::]) + np.array(_ttmp[::])) / 2)
                cv2.line(img, int2round(_tmp), int2round(_ttmp), (255, 255, 0), 2)
                cv2.putText(img, '%.3f' % (_length), int2round(_line_center), FONT_FACE, 0.4, (255, 255, 0), 1)
            elif _length < 250.0:
                _line_center = tuple((np.array(_tmp[::]) + np.array(_ttmp[::])) / 2)
                cv2.line(img, int2round(_tmp), int2round(_ttmp), (9, 249, 17), 2)
                cv2.putText(img, '%.3f' % (_length), int2round(_line_center), FONT_FACE, 0.4, (9, 249, 17), 1)
    if len(center_list) > 3:
        _spr_or_grp = _ccrowd_estimation_tune(center_list, PARAM_CROWD)
    else:
        _spr_or_grp = -1
    return img, np.array(_bbox_list), _spr_or_grp


def _ccrowd_estimation_tune(_center_list, PARAM_CROWD):
    _coc = np.mean(np.array(_center_list))
    _dist_avg = 0.0
    _count = 0
    for _tmp in _center_list:
        dist = np.linalg.norm(_coc - _tmp)
        if dist < PARAM_CROWD:
            _dist_avg += dist
            _count += 1
    if _count == 0:
        _dist_avg = -1
    else:
        _dist_avg = _dist_avg / _count
    return _dist_avg
