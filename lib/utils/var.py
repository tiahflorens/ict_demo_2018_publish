import os
import sys
import cv2
import getpass
#
if sys.platform == 'win32':
    PROJECT_PATH = 'D:\PycharmProjects\ict_demo_2018'
#     VIDEO_PATH = 'F:\DATASET/13.GIST/GIST2017'
elif sys.platform =='darwin':
    PROJECT_PATH = '/Users/peter/PycharmProjects/ict_demo_2018'
else:
    PROJECT_PATH = '/home/'+getpass.getuser() +'/extra/Workspace/ict_demo_2018'
#     # VIDEO_PATH = '/home/'+getpass.getuser() +'/Workspace/ict_demo_2017/videos'


ckpt_filename = os.path.join(PROJECT_PATH,'lib/models/ssd300/ssd_300_vgg.ckpt')
mlp_model_path = os.path.join(PROJECT_PATH, 'lib/models/ssd300/mlp_model.pickle')

GIST_VIDEO_PATH=os.path.join(PROJECT_PATH,'experiments/gist/video')
GIST_PATH=os.path.join(PROJECT_PATH,'experiments/gist')
GIST_LABEL_PATH=os.path.join(PROJECT_PATH,'experiments/gist/label')


rtsp_list2 = ['rtsp://admin:1234@172.26.19.157:6554/site1/video1',
             'rtsp://admin:1234@172.26.19.157:6554/site2/video1',
             'rtsp://admin:1234@172.26.19.157:6554/site3/video1',
             'rtsp://admin:1234@172.26.19.157:6554/site4/video1',
             'rtsp://admin:1234@172.26.19.157:6554/site5/video1',
             'rtsp://admin:1234@172.26.19.157:6554/site6/video1',
             'rtsp://admin:admin123-@qa.nexpa.co.kr:50017/stream2',
             'rtsp://admin:admin123-@qa.nexpa.co.kr:50020/stream2',
             'rtsp://admin:admin123-@qa.nexpa.co.kr:50018/stream2',
             'rtsp://admin:admin123-@qa.nexpa.co.kr:50019/stream2',
             'rtsp://admin:admin123-@qa.nexpa.co.kr:50022/stream2',
             'rtsp://admin:admin123-@qa.nexpa.co.kr:40024/stream2',
             'rtsp://admin:admin123-@qa.nexpa.co.kr:40025/stream2']

gist_list = ['rtsp://admin:gist2406@192.168.190.241',
             'rtsp://admin:gist2406@192.168.190.242',
             'rtsp://admin:gist2406@192.168.80.243',
             'rtsp://admin:gist2406@192.168.80.244']


nexpa_list = ['rtsp://admin:admin123-@qa.nexpa.co.kr:50017/stream2',
              'rtsp://admin:admin123-@qa.nexpa.co.kr:50020/stream2',
              # 'rtsp://admin:admin123-@qa.nexpa.co.kr:50018/stream2',
              'rtsp://admin:admin123-@qa.nexpa.co.kr:50019/stream2',
              'rtsp://admin:admin123-@qa.nexpa.co.kr:50022/stream2'
              # 'rtsp://admin:admin123-@qa.nexpa.co.kr:40024/stream2',
              # 'rtsp://admin:admin123-@qa.nexpa.co.kr:40025/stream2'
              ]


# gist_list = ['rtsp://admin:gist2406@192.168.190.241',
#              'rtsp://admin:gist2406@192.168.190.242',
#              'rtsp://admin:gist2406@192.168.80.241',
#              'rtsp://admin:gist2406@192.168.80.242',
#              'rtsp://admin:gist2406@192.168.80.243',
#              'rtsp://admin:gist2406@192.168.80.244']
#
# nexpa_list = ['rtsp://admin:admin123-@qa.nexpa.co.kr:50017/stream2',
#               'rtsp://admin:admin123-@qa.nexpa.co.kr:50020/stream2',
#               'rtsp://admin:admin123-@qa.nexpa.co.kr:50018/stream2',
#               'rtsp://admin:admin123-@qa.nexpa.co.kr:50019/stream2',
#               'rtsp://admin:admin123-@qa.nexpa.co.kr:50022/stream2',
#               'rtsp://admin:admin123-@qa.nexpa.co.kr:40024/stream2',
#               'rtsp://admin:admin123-@qa.nexpa.co.kr:40025/stream2']

rtsp_list = gist_list + nexpa_list

FONT_FACE = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 1.5
FONT_THINKESS = 2
# DP_WIDTH = 440
# DP_HEIGHT = 300
# DP_ROW = 4

DP_WIDTH = 1280
DP_HEIGHT = 720
DP_ROW = 3

FRAME_COUNT_LOC = (30, 50)
FRAME_DIST_LOC = (990,50)
FRAME_STATUS_LOC = (410,50)

# DP_WIDTH = 640
# DP_HEIGHT = 480
DP_ROW = 3

THICKNESS = 2

RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
ORANGE = (255, 140, 0)
YELLOW = (255,255,0)
MAGENTA = (255,0,255)
LIGHT_GREEN = (9,249,17)
BALCK =(0,0,0)
WHITE= (255,255,255)

DP_YES ='yes'
DP_NO ='no'

ARG_IN_VIDEO ='video'
ARG_IN_STREAM ='stream'

ARG_OUT_NONE ='none'
ARG_OUT_VIDEO ='video'
ARG_OUT_IMAGE ='image'

COND_DIST_NORMAL = -1
COND_DIST_GROUP = 180
COND_DIST_SPRADING = 240

DEBUG= 'debug'
INFO ='info'
WARNING='warning'