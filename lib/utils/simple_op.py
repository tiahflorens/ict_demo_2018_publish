
import pickle
import threading
import time

import numpy as np

from lib.utils.var import *
import lib.utils.var

def isEmpty(q, l):
    l.acquire()
    b = q.empty()
    l.release()
    return b


def pux(q, l, x):
    l.acquire()
    b = q.put(x)
    l.release()
    return b


def popx(q, l):
    l.acquire()
    b = q.get()
    l.release()
    return b


def qsize_(q, l):
    l.acquire()
    b = q.qsize()
    l.release()
    return b


def velocity(points):
    _distance = 0.0
    if len(points) > 3:
        for x in range(len(points) - 1):
            dist = np.linalg.norm(points[x] - points[x + 1])
            _distance += dist
        _vel = _distance / float(len(points) - 1)
    else:
        _vel = -1
    return _vel


def load_mlp_model():

    with open(mlp_model_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        mlp_model = u.load()
        return mlp_model


def get_window_loc(row, idx):
    y, x = divmod(idx, row)


    xx = (x * DP_WIDTH) +80
    yy = (y * DP_HEIGHT) +90

    if xx == 70:
        xx = 0
    if yy== 0:
        yy =0

    return xx, yy



def read_video_as_list_by_path(path, fname_ext = None ,color_flag=1, print_flag= 0):
    """

    :param path: path to video file
    :param fname: video file name
    :return: image list
    """

    cap = cv2.VideoCapture()
    if fname_ext is None:
        cap.open(path)
    else:
        cap.open(path + '/' + fname_ext)


    if not cap.isOpened():
        if fname_ext is None:
            print('not exist ', path)
        else:
            print('not exist ', path+'/'+fname_ext)
        quit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    props = { 'fps':int(fps), 'fourcc': int(fourcc), 'width': int(width), 'height': int(height), 'length':int(length)}
    f = []
    while 1:
        ret, frame = cap.read()
        if ret is False:
            break
        if color_flag:
            f.append(frame)
        else:
            f.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    if len(f) is 0:
        print('tools, read video as list by path : no frames in video')
        quit()

    cap.release()

    if print_flag:
        print('Video size: ', len(f), ' params: ' ,(fps,width,height) )

    return f, props


def read_video_by_path(path , fname , isprint=0):
    """

    :param path: path to video file
    :param fname: video file name
    :return: image list
    """
    print("video reading from " , path + '/' + fname)

    cap = cv2.VideoCapture()
    cap.open(path + '/' + fname)


    if not cap.isOpened():
        print ('tools,', path + '/' + fname,  'file not exist')
        quit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    props = { 'fps':int(fps), 'fourcc': fourcc, 'width': int(width), 'height': int(height), 'length':int(length)}
    if isprint:
        print(props)
    return cap , props



def proper_frames_fps(diff):
    """
    Compute repeat time to make video as online playing.

    :Make duplicated frames when writing a video,
    since video writer has fixed fps.

    :param diff: time difference between start and end time of algorithm.
    :return:
    """
    fps = 1/diff
    fps = int(fps)
    if fps > 30:
        repeat = 1
    elif fps ==0:
        repeat = 1
    else:
        repeat = int(30/fps)

    return fps, min(repeat,30)


def texting_frame(frame, count, fps, obj_label, color_):
    count_msg = '%-10s' % ('frame: %d' % (count))
    cv2.putText(frame, count_msg, FRAME_COUNT_LOC, FONT_FACE, FONT_SCALE, RED, THICKNESS)  # frame

    txtSize, baseLine = cv2.getTextSize(count_msg, FONT_FACE, FONT_SCALE, THICKNESS)
    fps_loc = (30, FRAME_COUNT_LOC[1] + txtSize[1] + 15)

    fps_msg = '%-10s' % ('fps: %d' % (fps))
    cv2.putText(frame, fps_msg, fps_loc, FONT_FACE, FONT_SCALE, (255, 255, 255), THICKNESS)

    txtSize, baseLine = cv2.getTextSize(fps_msg, FONT_FACE, FONT_SCALE, THICKNESS)
    obj_loc = (30, fps_loc[1] + txtSize[1] + 15)
    cv2.putText(frame, '%-10s' % obj_label, obj_loc, FONT_FACE, FONT_SCALE, color_, THICKNESS)
    return frame




def get_obj_level_label(_dist_btw_objects):
    # 0 normal
    # 1 spreading
    # 2 grouping
    # 3 assult
    # 4 kidnap

    print(COND_DIST_NORMAL, COND_DIST_GROUP, COND_DIST_SPRADING)
    print('--------------------------------')
    if _dist_btw_objects == COND_DIST_NORMAL:
        msg, color_, flag = 'Normal', GREEN , [1,0,0,0,0]
    elif _dist_btw_objects < COND_DIST_GROUP:
        msg, color_ , flag  = 'Grouping', RED , [0,0,1,0,0]
    elif _dist_btw_objects > COND_DIST_SPRADING:
        msg, color_ , flag = 'Spreading', RED, [0,1,0,0,0]
    else:
        msg, color_ , flag = '', RED , [0,0,0,0,0]

    return msg, color_, flag


def get_obj_level_label_tuning(_dist_btw_objects, G, S):
    # 0 normal
    # 1 spreading
    # 2 grouping
    # 3 assult
    # 4 kidnap

    if _dist_btw_objects == -1 :
        msg, color_, flag = 'Normal', GREEN , [1,0,0,0,0]
    elif _dist_btw_objects < G:
        msg, color_ , flag  = 'Grouping', RED , [0,0,1,0,0]
    elif _dist_btw_objects > S:
        msg, color_ , flag = 'Spreading', RED, [0,1,0,0,0]
    else:
        msg, color_ , flag = '', RED , [0,0,0,0,0]

    return msg, color_, flag




def draw_trajectory(_track_list):
    _list = []
    _idlist = []
    _init = 0
    _trajectories = []
    _return_set = []
    for _track in _track_list:
        for d in _track:
            d = d.astype(np.int32)
            _id = d[4]
            if (_id in _idlist) == False:
                _idlist.append(_id)
            _center = [(int(d[0]) + int(d[2])) / 2, (int(d[1]) + int(d[3])) / 2]
            _list.append([_id, _center])
    for i in range(len(_idlist)):
        _trajectories.append([])

    _idcount = 0
    for _tid in _idlist:
        for _tmp in _list:
            if _tmp[0] == _tid:
                _trajectories[_idcount].append(_tmp[1])
        _return_set.append([_tid, _trajectories])
        _idcount += 1
    return _track_list[1:len(_track_list)], _return_set
