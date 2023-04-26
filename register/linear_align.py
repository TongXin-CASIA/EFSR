import cv2
import numpy as np


def affine_align(img_r, img_m):
    """
    params: img_r, img_m
    """
    # estimate the affine transformation
    sift = cv2.SIFT_create(5000)
    kp_r, des_r = sift.detectAndCompute(img_r, None)
    kp_m, des_m = sift.detectAndCompute(img_m, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_m, des_r)
    matches = sorted(matches, key=lambda x: x.distance)[: 2000]
    src_pts = np.float32([kp_m[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_r[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffine2D(src_pts, dst_pts, None, cv2.RANSAC, 5.0)
    # warp the image
    return M, mask.sum()


def warp_affine(img_m, M):
    """
    params: img_m
    params: M
    """
    return cv2.warpAffine(img_m, M, (img_m.shape[1], img_m.shape[0]))