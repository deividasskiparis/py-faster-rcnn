#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""


import os, sys, cv2
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__),"../caffe-fast-rcnn/python")))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__),"../tools")))
import _init_paths
import caffe
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from cv2 import ml
import argparse
from classes import Group, Person
from get_feature_vector import get_feature_vector
from sklearn import svm
from numpy.linalg import norm

sys.path.append(__file__)

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


WINDOW_NAME = "PEOPLE DETECTIONS"


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args


def detect_people(net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    # im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    # timer = Timer()
    # timer.tic()
    scores, boxes = im_detect(net, im)


    # timer.toc()
    # print ('Detection took {:.3f}s for '
    #        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    # for cls_ind, cls in enumerate(CLASSES[1:]):

    cls_ind, cls = 15, 'Person'
    # cls_ind += 1 # because we skipped background
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]

    # print dets2
    # vis_detections_webcam(im, cls, dets, thresh=CONF_THRESH)


    return dets


def vis_detections_webcam(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]


    # im = im[:, :, (2, 1, 0)]
    text_no_people = 'No of people: %d' % len(inds)
    cv2.putText(im, text_no_people, org=(10, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1, color=(255, 255, 255))
    if len(inds) == 0:
        return
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), thickness=2)

        text_write = '%s %0.3f'%(class_name, score)
        cv2.putText(im, text_write, org=(int(bbox[0]), int(bbox[1]) - 2), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 255, 255))

        cv2.imshow(WINDOW_NAME, im)


def vis_people(im, list_of_people, total_count=0):

    assert isinstance(list_of_people, list)

    cv2.rectangle(im, (0, 0), (im.shape[1], 35), (255, 255, 255), -1)

    text_no_people = 'On-screen: %d    Total: %d' % (len(list_of_people), total_count)
    cv2.putText(im, text_no_people, org=(10, 25), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1, color=(0, 0, 0))
    if len(list_of_people) == 0:
        return
    for p in list_of_people:
        assert isinstance(p, Person)

        actual_pos = p.fv[-1, :3]
        pred_pos = p.next_position
        bbox = p.bbox

        actual_color = (0, 255, 0)
        if p.last_seen > 0:
            actual_color = (0, 0, 255)
        pred_color = (255, 0, 0)

        cv2.circle(im, (int(actual_pos[0]), int(actual_pos[1])), int(30000/max(actual_pos[2], 1000)), actual_color, thickness=2)
        if pred_pos is not None:
            cv2.circle(im, (int(pred_pos[0][0]), int(pred_pos[0][1])), int(30000/max(pred_pos[0][2], 1000)), pred_color, thickness=2)

        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), thickness=2)

        text_write = '%d' % p.personID
        cv2.putText(im, text_write, org=(int(bbox[0]), int(bbox[1]) - 2), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 255, 0))


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    # Initialize camera
    cap = cv2.VideoCapture(cv2.CAP_OPENNI2)
    assert cap.isOpened()
    FRAME_W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    FRAME_H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print "FRAME W x H = %d x %d" % (FRAME_W, FRAME_H)

    # Initialize the detector
    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Parameters
    CONF_THRESH = 0.9       # Det:  Person detection confidence for Faster-RCNN
    TH_MAX_DEPTH = 3000     # Det:  Maximum distance at which the people are detected
    TH_XY = 1.5             # ReId: Threshold for X,Y displacement if less than (prediction - detection)/bbox_width
    TH_D = 100              # ReId: Max difference in depth
    TH_LAST_SEEN = 30       # ReId: Delete the person if he was last seen X frames ago
    TH_TEMP_LAST_SEEN = 3   # ReId: Remove people from temporary buffer is unseen for X frames
    TH_PERSISTENCE = 3      # ReId: Min number of re-appearances before a detection is treated as a person
    LAST_N_FVS = 5          # Number of latest feature vectors to take into account for classification
    TH_MAX_DIS = 0.25       # Maximum dissimilarity between detections to be matched

    # Initial variables
    people_counter = 0      # Total people count
    people_on_screen = 0    # Number of people on screen
    current_people = []     # List of 'Current People' - temporary people which persisted for more than TH_PERSISTENCE
    temp_buffer = []        # List of temporary people

    while cap.grab():

        # Get RGB and Depth frames
        okay1, depth_map = cap.retrieve(0, cv2.CAP_OPENNI_DEPTH_MAP)
        okay2, rgb_image = cap.retrieve(0, cv2.CAP_OPENNI_BGR_IMAGE)

        assert okay1 and okay2

        # Convert depth in uint8 for visualization purposes
        depth_map2 = depth_map.copy().astype(np.float) / 10000 * 255
        depth_map2 = depth_map2.astype(np.uint8)

        # Detect all people in a frame
        proposals = detect_people(net, rgb_image)
        proposals = proposals[np.where(proposals[:, -1] >= CONF_THRESH)]

        # For all detections, generate feature vector for re-identification
        detection_fvs = get_feature_vector(rgb_image, depth_map, proposals)

        # Ensure the detections are not over the maximum depth
        detection_fvs = detection_fvs[np.where(detection_fvs[:, 2] < TH_MAX_DEPTH)]

        detections = []
        no_people = detection_fvs.shape[0]

        # Process proposals from detector
        for i in range(no_people):  # For every detected person
            bbox = proposals[i, :4] # get bounding box
            fv = detection_fvs[i]   # get feature vector
            p = Person(bbox=bbox, feat_vec=fv)  # Record as a person
            detections.append(p)    # Add to detections

        ################### FOR FUTURE - TRAINING SUPERVISED CLASSIFIERS ###################
        # Get a list of feature vector for 'Current People'
        current_fvs = np.ndarray(shape=(0, detection_fvs.shape[1]), dtype=np.float)
        current_lbls = np.ndarray(shape=(0,), dtype=np.int)

        for idx, g in enumerate(current_people):
            assert isinstance(g, Person)
            mnm = min(g.fv.shape[0], LAST_N_FVS)
            current_fvs = np.append(current_fvs, g.fv[-mnm:], axis=0)
            current_lbls = np.append(current_lbls, np.tile(np.array([g.personID]), mnm))
        cluster_n = len(current_people)

        # Can be SVM or any other
        clf = svm.SVC(decision_function_shape='ovr')

        ###########################################################################

        # Using Kalman, predict next location of all current people and try to match with detections
        for idx, cp in enumerate(current_people):   # For every person in the 'Current People'
            assert isinstance(cp, Person)
            next_position = cp.predict_next()   # Predict next position based on previous motion

            if next_position is None:
                # If the person is new, no previous data will be present to predict, so simply copy the current data
                next_position = cp.fv[-1, :3]

            if len(detections) == 0:    # No detections or all have been assigned
                # Treat person as occluded and record predicted data
                temp_v = np.ndarray(shape=(detection_fvs.shape[1]), dtype=np.float)
                temp_v[:3] = next_position
                temp_v[3:] = cp.fv[-1, 3:]

                current_people[idx] += np.array([temp_v])
                current_people[idx].last_seen += 1
                current_people[idx].visibility.append(False)
                continue

            # Calculate differences in XY and Depth for comparison
            distancesXY = norm(detection_fvs[:, :2] - np.tile(next_position[:, :2], (detection_fvs.shape[0], 1)),
                               ord=2, axis=1, keepdims=True)\
                        /(cp.bbox[2]-cp.bbox[0])  # dividing by bbox width, gives distance, relative to size
            distancesD = np.array([detection_fvs[:, 2] - np.tile(next_position[-1, 2], (detection_fvs.shape[0]))]).T

            # Get subset of detections which conform with thresholds
            close_inds = np.where((distancesXY < TH_XY) & (distancesD < TH_D))[0]

            if close_inds.shape[0] == 0:    # In case there are none, which satisfy conditions
                # Treat person as occluded and record predicted data
                temp_v = np.ndarray(shape=(detection_fvs.shape[1]), dtype=np.float)
                temp_v[:3] = next_position
                temp_v[3:] = cp.fv[-1, 3:]
                current_people[idx] += np.array([temp_v])
                current_people[idx].last_seen += 1
                current_people[idx].visibility.append(False)

            else:   # There are detections satisfying the condition(s)
                # Get best match - Closest
                # best_ind = np.argmin(distancesXY[close_inds])
                # best_ind = close_inds[best_ind]

                # Get best match - most similar
                mnm = min(cp.fv.shape[0], LAST_N_FVS)
                similarity = norm(detection_fvs[close_inds, 3:] - np.tile(np.average(cp.fv[-mnm:, 3:], axis=0),
                                                                 (close_inds.shape[0], 1)),
                                   ord=2, axis=1, keepdims=True)

                simil_inds = np.where(similarity < TH_MAX_DIS)[0]

                if simil_inds.shape[0] == 0:
                    continue
                best_ind = np.argmin(similarity[simil_inds])
                best_ind = close_inds[simil_inds[best_ind]]

                print "Similarity:", similarity
                print "Best_ind:", best_ind
                # Merge detection with current person and remove
                current_people[idx] += detections.pop(best_ind)

                # Remove the detections feature vector
                detection_fvs = np.delete(detection_fvs, best_ind, axis=0)

                # Reset Person's statistics
                current_people[idx].last_seen = 0
                current_people[idx].appearances += 1
                current_people[idx].visibility.append(True)

        # Do exactly the same for the temp_buffer
        temp_buffer_fvs = np.ndarray(shape=(0, detection_fvs.shape[1]), dtype=np.float)
        temp_buffer_lbls = np.ndarray(shape=(0), dtype=np.float)

        for idx, tb in enumerate(temp_buffer):  # For every person in the temp buffer
            assert isinstance(tb, Person)
            next_position = tb.predict_next()   # Predict next position based on previous motion

            if len(detections) == 0:    # No detections or all have been assigned

                # Treat person as occluded and record predicted data
                temp_v = np.ndarray(shape=(detection_fvs.shape[1]), dtype=np.float)
                temp_v[:3] = next_position
                temp_v[3:] = tb.fv[-1, 3:]

                temp_buffer[idx] += np.array([temp_v])
                temp_buffer[idx].last_seen += 1
                temp_buffer[idx].visibility.append(False)

                continue

            # Calculate differences in XY and Depth for comparison
            distancesXY = norm(detection_fvs[:, :2] - np.tile(next_position[:, :2], (detection_fvs.shape[0], 1)), ord=2, axis=1, keepdims=True)/(tb.bbox[2]-tb.bbox[0])
            distancesD = np.array([detection_fvs[:, 2] - np.tile(tb.fv[-1, 2], (detection_fvs.shape[0]))]).T

            # Get subset of people which conform with thresholds
            close_inds = np.where((distancesXY < TH_XY) & (distancesD < TH_D))[0]

            if close_inds.shape[0] == 0:    # In case there are none, which satisfy conditions
                # Treat person as occluded and record predicted data
                temp_v = np.ndarray(shape=(detection_fvs.shape[1]), dtype=np.float)
                temp_v[:3] = next_position
                temp_v[3:] = tb.fv[-1, 3:]

                temp_buffer[idx] += np.array([temp_v])
                temp_buffer[idx].last_seen += 1
                temp_buffer[idx].visibility.append(False)
            else:   # There are detections satisfying the condition(s)
                # Get best match - Closest
                # min_ind = np.argmin(distancesXY[close_inds])
                # min_ind = close_inds[min_ind]
                #

                # Get best match - most similar
                mnm = min(tb.fv.shape[0], LAST_N_FVS)
                similarity = norm(detection_fvs[close_inds, 3:] - np.tile(np.average(tb.fv[-mnm:, 3:], axis=0),
                                                                 (close_inds.shape[0], 1)),
                                   ord=2, axis=1, keepdims=True)

                simil_inds = np.where(similarity < TH_MAX_DIS)[0]
                if simil_inds.shape[0] == 0:
                    continue

                best_ind = np.argmin(similarity[simil_inds])
                best_ind = close_inds[simil_inds[best_ind]]

                # Merge detection with current person and remove
                temp_buffer[idx] += detections.pop(best_ind)

                # Remove the detections feature vector
                detection_fvs = np.delete(detection_fvs, best_ind, axis=0)

                # Reset Person's statistics
                temp_buffer[idx].last_seen = 0
                temp_buffer[idx].appearances += 1
                temp_buffer[idx].visibility.append(True)

        # Everything that is left, is added as new people
        temp_buffer += detections

        # Clean up temp_buffer
        for idx, q in enumerate(temp_buffer):
            if q.last_seen > TH_TEMP_LAST_SEEN: # Over the last-seen threshold
                # Mark for removal
                temp_buffer[idx] = None
            elif q.appearances > TH_PERSISTENCE and q.last_seen == 0:   # Over the persistence threshold
                # Add to 'current_people'
                q.personID = people_counter
                current_people.append(q)
                temp_buffer[idx] = None
                print "New person added ID = %d" % people_counter
                people_counter += 1

        # Get rid marked People
        temp_buffer = [x for x in temp_buffer if x is not None]

        # Clean-up 'current_people'

        for idx, w in enumerate(current_people):
            if idx == 0:
                print "Current people"
            assert isinstance(w, Person)
            print "\t iD: %d, last_seen = %d, predicted = %s" % (w.personID, w.last_seen, w.next_position)
            if w.last_seen > TH_LAST_SEEN:  # Over the last-seen threshold
                # Mark for removal
                current_people[idx] = None

            # If the person is occluded, check if his next predicted position is outside of screen (person has left)
            elif \
            w.next_position is not None and w.last_seen > 0\
            and ((w.next_position[0][0] > FRAME_W or w.next_position[0][0] < 0)
            or (w.next_position[0][1] > FRAME_H or w.next_position[0][1] < 0)):
                # Mark for removal
                current_people[idx] = None

        # Get rid marked People
        current_people = [x for x in current_people if x is not None]

        # Draw visualizations of 'current_people'
        vis_people(rgb_image, current_people, people_counter)

        # Display
        cv2.imshow("depth map2", depth_map2)
        cv2.imshow(WINDOW_NAME, rgb_image)
        key = cv2.waitKey(33)
        if key == 27:
            break