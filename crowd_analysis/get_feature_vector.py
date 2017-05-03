import numpy as np
import cv2


def get_feature_vector(img, depth, bboxes, histbin=32):

    assert (img.shape[2] == 3)
    assert (len(depth.shape) == 2)
    assert (img.shape[0:2] == depth.shape[0:2])

    bboxes = bboxes.astype(np.int)
    feature_vectors = np.ndarray(shape=(bboxes.shape[0], 3+histbin), dtype=np.float)
    for b_id, bbox in enumerate(bboxes):
        fv = np.ndarray(feature_vectors.shape[1], dtype=np.float)

        ### Get mask for depth
        # Some ref values
        x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]

        # centre point of bbox and corresponding depth value
        cp_bbox = (int(x + w/2), int(y))
        depth_ref = depth[cp_bbox[1], cp_bbox[0]]

        # threshold the depth image at the bounding box
        subim_D = depth[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        subim_C = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        mask = cv2.inRange(subim_D, depth_ref - 150, depth_ref + 150)

        # Record
        fv[0:3] = [cp_bbox[0], cp_bbox[1], depth_ref]

        ### Get histograms
        # Params
        channels = [0]
        hist_size = [histbin]
        ranges = [0, 255]

        # cv2.imshow("col_sub", subim_C)
        # cv2.imshow("col_dep", subim_D)
        # cv2.imshow("mask", mask)

        for ch in range(img.shape[2]): #for every channel in an image
            # Get histogram
            v = cv2.calcHist([subim_C[:, :, ch]], channels, mask, hist_size, ranges)
            # Record
            fv[3:] = v.T / np.sum(v)

        feature_vectors[b_id, :] = fv

    return feature_vectors