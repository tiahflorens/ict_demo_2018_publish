import argparse

parser = argparse.ArgumentParser(description='This is a ICT demo program')
parser.add_argument('--version', action='version', version='%(prog)s 1.0')

parser.add_argument('--display', '-dp', dest='dp', action="store", help=' [yes | no] display option yes/no.',
                    default='yes', type=str)
parser.add_argument('--videopath', '-path', dest='path', action="store", help=' give video path under project path',
                    type=str)

results = parser.parse_args()
print(results)

import os, sys, datetime, cv2

sys.path.append(os.getcwd())

from lib.utils import sort
from lib.utils.simple_op import *
from lib.utils.core import process_image, process_ssd_result
import matplotlib.pyplot as plt
import lib.utils.var as var


mot_tracker = sort.Sort()
mlp_model = load_mlp_model()
stamp_list = [[], [], [], []]

untracked_iou_param_set = {'bag2': 0.2, 'fight1': 0.1, 'car_person': 0.1}
assign_iou_param_set = {'bag2': 0.5, 'fight1': 0.3, 'car_person': 0.3}


# np.seterr(all='print')
delay = 3
untracked_iou_threshold = 0.2
assing_iou_threshold = 0.3



def make_result_video(video_path, _video_name, now):
    print("Starting " + _video_name)
    count, dist = 0, 0
    unlabeled_counter = 753

    tracking_set = {}

    cap, prop = read_video_by_path(video_path, _video_name)

    n_box_list, dist_list = [], []
    video_name = _video_name.split('.')[0]

    DP_FRAME_SIZE = (prop['width'], prop['height'])

    ROIs = roi_center(DP_FRAME_SIZE, 90)
    logger.info('Frmae size is ', DP_FRAME_SIZE)

    out_home_dir_path = os.path.join(var.GIST_PATH, 'output')
    out_this_dir_path = os.path.join(out_home_dir_path, video_name)
    # print('Video to ', out_this_dir_path)


    if not os.path.exists(out_this_dir_path):
        os.makedirs(out_this_dir_path)

    video_path = os.path.join(out_this_dir_path + '/' + now + '_' + video_name + '.avi')
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    writer = cv2.VideoWriter(video_path, fourcc, 30, DP_FRAME_SIZE)

    while 1:
        start_timestamp = time.time()

        ret, im = cap.read()
        if ret is False:
            break

        count += 1
        count_msg = '%-10s' % ('Frame:: %d' % (count))
        cv2.putText(im, count_msg, FRAME_COUNT_LOC, cv2.FONT_HERSHEY_DUPLEX, FONT_SCALE, WHITE, FONT_THINKESS)  # point is left-bottom

        cv2.rectangle(im, ROIs[0], ROIs[1], ORANGE, 2)


        ssd_stime = time.time()
        rclasses, rscores, rbboxes = process_image(im)
        took_sdd_time = time.time() - ssd_stime

        visual_stime = time.time()

        _bbox_list, thing_list = process_ssd_result(rclasses, rbboxes, im.shape)
        # case1: x1,y1,x2,y2

        took_visual_time = time.time() - visual_stime

        add_counter_trackset(tracking_set)
        if len(_bbox_list) > 0:
            bbox_list = convert_bbox_to_list(_bbox_list)

            trackers = mot_tracker.update(np.array(bbox_list))
            unlabeled_counter = assign_tracker_id_bbox(bbox_list, trackers, unlabeled_counter, assing_iou_threshold)


            restore_bbox22(tracking_set, bbox_list, ROIs, untracked_iou_threshold)

            tracking_list = convert_set_to_list(tracking_set)

            centers = convert_list_ellipse_centers(bbox_list)
            draw_centers(im, centers)

            for box in bbox_list:
                cx = int((box[0] + box[2]) / 2)
                center = (cx, box[3])
                ax1, ax2 = int((box[2] - box[0]) / 2), 20
                axes = (ax1, ax2)
                angle = 0  # rotation
                cv2.ellipse(im, center, axes, angle, 0, 360, LIGHT_GREEN, 3)

            label_list = ['PERSON' for x in range(len(bbox_list))]
            # drawer.labeling_bbox_list(im,tracking_list, 1, label_list)

            if len(tracking_list) > 1:
                dist_mat = calculate_distance(centers)
                draw_lines(im, dist_mat)
                dist = average_distance(dist_mat)
        del_counter_trackset(tracking_set)
        # cv2.imwrite(os.path.join(out_this_dir_path, video_name + str(count).zfill(6) + '.png'), im)

        n_box_list.append(len(tracking_set.keys()))
        dist_list.append(dist)
        writer.write(im)

        # writer.write(frame)

        # stamp_list[0].append(took_sdd_time)
        # stamp_list[1].append(took_visual_time)
        # stamp_list[2].append(took_track_time)
        # stamp_list[3].append(len(bbox_list))

        if results.dp == 'yes':
            cv2.imshow('dafd', im)
            if cv2.waitKey(1) & 0xff == 27:
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()
    # writer.release()
    # display_timemstamp(out_this_dir_path, now, video_name)
    # plot_history(dist_list,n_box_list,video_name)

    print("Exiting dp")


def plot_history(dist_list, n_box_list, video_name):
    plt.clf()
    plt.cla()
    # gt_path = os.path.join(var.PROJECT_PATH, 'experiments/gist/label', video_name + '_label')
    # gt_list = pickle.load(open(gt_path, 'rb'))
    # gts = np.argmax(gt_list, axis=1)

    print(len(dist_list), len(n_box_list))
    X = np.arange(len(dist_list))
    plt.title("History")
    ax = plt.subplot(2, 1, 1)
    ax.plot(X, dist_list, 'r-', label='distance')

    ax.legend(loc='best')

    ax = plt.subplot(2, 1, 2)
    ax.plot(X, n_box_list, 'b-', label='# of bboxes')
    ax.legend(loc='best')
    n_min = np.min(n_box_list)
    n_max = np.max(n_box_list)
    ax.set_ylim([n_min - 1, n_max + 1])

    plt.title(video_name)
    plt.tight_layout()
    plt.show()


def display_timemstamp(path, now, video_name):
    plt.clf()
    plt.cla()

    sumed_time = np.array((stamp_list[0], stamp_list[1], stamp_list[2])).sum(axis=0)

    X = np.arange(len(stamp_list[0]))
    plt.title("Time stamp")
    ax = plt.subplot(2, 1, 1)
    ax.plot(X, stamp_list[3], 'r-', label='bbox')
    ax.legend(loc='best')

    ax = plt.subplot(2, 1, 2)
    ax.plot(X, stamp_list[0], 'b-', label='ssd time')
    ax.plot(X, stamp_list[1], 'r-', label='visual time')
    ax.plot(X, stamp_list[2], 'g-', label='track time')
    ax.plot(X, sumed_time, 'y--', label='sum time')
    ax.legend(loc='best')
    ax.set_ylim([0, 0.2])

    fig_path = os.path.join(path + '/' + now + '_' + video_name + '.png')
    plt.savefig(fig_path)
    # plt.show()


def convert_set_to_list(givenset):
    a = []
    for k in givenset.keys():
        a.append(givenset[k])
    return a


def roi_center(FRAME_SIZE, RATIO):
    w, h = FRAME_SIZE[0], FRAME_SIZE[1]
    _w, _h = int(w * RATIO / 100), int(h * RATIO / 100)

    xres = w - _w
    hres = h - _h
    x1, x2 = xres, w - xres
    y1, y2 = hres, h - hres

    return (x1, y1), (x2, y2)


def convert_bbox_to_list(_bbox_list):
    bbox_list = []
    for bbox in _bbox_list:
        bbox = np.array(bbox, dtype=np.uint32)
        bbox_list.append(list(bbox))
    return bbox_list


def add_counter_trackset(tracking_set):
    for k in tracking_set.keys():
        bbox = tracking_set[k]
        bbox[5] += 1


def del_counter_trackset(tracking_set):
    poplists = []
    for k in tracking_set.keys():
        bbox = tracking_set[k]
        if bbox[5] > 40:
            poplists.append(k)

    for p in poplists:
        tracking_set.pop(p)


def restore_bbox22(tracking_set, curr_bbox_list, ROIs, iou_threshold):
    updated_key_list = []
    untracked_bbox_list = []

    for cbbox in curr_bbox_list:  # Id matching first
        key = cbbox[4]
        if (key in tracking_set.keys()) and key != 0:
            cbbox[5] = 0  # init
            tracking_set[key] = cbbox
            updated_key_list.append(key)
        else:
            untracked_bbox_list.append(cbbox)

    for ubbox in untracked_bbox_list:  # with IOU, tracking objects which are unlabeled.
        iou_list, key_list = [], []
        for key in tracking_set.keys():
            if key not in updated_key_list:
                bbox = tracking_set[key]
                iou_val = iou(bbox, ubbox)
                iou_list.append(iou_val)
                key_list.append(key)


        if len(iou_list) > 0:
            max_iou_idx = np.argmax(iou_list, axis=0)
            max_iou_val = iou_list[max_iou_idx]
            max_iou_key = key_list[max_iou_idx]
            if max_iou_val > iou_threshold:  # update key value to latest.
                # tracking_set.pop(ubbox[4])
                updated_key_list.append(ubbox[4])
                """
                # before update ubboxkey, orginal key is add to updated_list, 
                to prevent accidentally adding untracked bbox in the next step.
                """

                ubbox[4] = max_iou_key  # updated ubbox to existing key.
                ubbox[5] = 0  # init
                tracking_set[max_iou_key] = ubbox  # update bbox info

    for ubbox in untracked_bbox_list:
        key = ubbox[4]  # 2
        if key not in updated_key_list:
            tracking_set[key] = ubbox

    pop_key_list = []
    for key in tracking_set.keys():
        center = get_bbox_center(tracking_set[key])
        if not check_center_in_roi(center, ROIs):
            pop_key_list.append(key)

    for key in pop_key_list:
        tracking_set.pop(key)

def check_center_in_roi(center, ROIs):
    p1, p2 = ROIs
    if (p1[0] < center[0] < p2[0]) and (p1[1] < center[1] < p2[1]):
        return True
    else:
        return False


def assign_tracker_id_bbox(_bbox_list, tracker_list, counter, iou_threshold):
    for bbox in _bbox_list:
        bbox.append(0)  # Adding counter
        for _track in tracker_list:
            track = _track.astype(np.uint32)
            a = iou(track, bbox)
            if a > iou_threshold:
                bbox[4] = int(track[4])
        if bbox[4] == 0:
            bbox[4] = counter
            counter += 1
    return counter


def convert_list_ellipse_centers(bbox_list):
    """

    :param bbox_list:
    :return: [ (cx, cy) ... ]
    """
    center_list = []
    for bbox in bbox_list:
        cx = int((bbox[0] + bbox[2]) / 2)
        center_list.append((cx, bbox[3]))
    return center_list


def convert_list_bbox_centers(bbox_list):
    """

    :param bbox_list:
    :return: [ (cx, cy) ... ]
    """
    center_list = []
    for bbox in bbox_list:
        cx, cy = get_bbox_center(bbox)
        center_list.append((cx, cy))
    return center_list


def get_bbox_center(bbox):
    cx = int((bbox[0] + bbox[2]) / 2)
    cy = int((bbox[1] + bbox[3]) / 2)
    return cx, cy


def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """

    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / (
            (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o)


def draw_lines(frame, dist_mat):
    for dset in dist_mat:
        p1 = dset[0]
        p2 = dset[1]
        cv2.line(frame, p1, p2, GREEN, 3)

        dist = str(dset[2])
        tx = int((p1[0] + p2[0]) / 2)
        ty = int((p1[1] + p2[1]) / 2)
        cv2.putText(frame, dist, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.5, ORANGE, 2)


def draw_centers(frame, centers):
    for center in centers:
        cv2.circle(frame, center, 10, RED, -1)


def calculate_distance(_center_list):
    """
    select two centers in a single list.
    Unique pair.

    :param _center_list:
    :return:  ( c1, c2, dist )
    """
    center_np = np.array(_center_list)
    dist_list = []
    n = len(_center_list)
    for i in range(n):
        nn = i + 1
        for j in range(nn, n):
            a = center_np[i]
            b = center_np[j]
            d = int(np.linalg.norm(a - b))
            c = _center_list[i]
            e = _center_list[j]
            dist_list.append((c, e, d))
    return dist_list


def average_distance(dist_mat):
    a = []
    for tmp in dist_mat:
        a.append(tmp[2])
    return np.mean(a)


def run_demo(args):
    args.path = GIST_VIDEO_PATH
    now = datetime.datetime.now().strftime('%m.%d_%H.%M.%S')
    # args.path =os.path.join(PROJECT_PATH,'experiments/gist/video')
    # args.path = os.path.join('C://Users/peter/Downloads/nexpa/all')
    args.path = os.path.join('/home/peter/extra/Workspace/ict_demo_2018/experiments/gist/video')
    logger.info('given path :' + args.path)
    for aa in os.listdir(args.path)[0:1]:

        print(args.path + ' ' + aa)
        if os.path.isdir(os.path.join(args.path, aa)):
            continue
        if aa.split('.')[1] in ['ini']:
            continue
        make_result_video(args.path, aa, now)
        # quit()

        logger.info("system exit")


run_demo(results)
