import os
import cv2
import glob
import shutil
import argparse
import numpy as np
from PIL import Image
from Experiment.estimate import compute_ncc, compute_dice_coefficients, compute_dice_max_k
from register.non_linear_align import FlowAligner


def pairwise_align(img_r, img_m, aligner, label=None):
    field = aligner.generate_field(img_r, img_m)
    img_w = aligner.warp_with_field(img_m, field.clone())
    if label is not None:
        label = aligner.warp_with_field(label, field.clone(), "nearest")
        return img_w, label
    else:
        return img_w


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, default="test1.png")
    parser.add_argument("--moving", type=str, default="test2.png")
    parser.add_argument("--model_path", type=str, default='../models/pth')
    parser.add_argument("--output", type=str, default='warped.png')
    args = parser.parse_args()
    aligner = FlowAligner(args.model_path, True, "cuda")
    img_r = np.array(Image.open(args.reference))
    img_m = np.array(Image.open(args.moving))
    img_w = pairwise_align(img_r, img_m, aligner)
    cv2.imwrite(args.output, img_w)