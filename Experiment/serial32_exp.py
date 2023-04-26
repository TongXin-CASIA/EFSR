import os
import cv2
import sys

import scipy.io

# Add the directory containing the Experiment package to the PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
sys.path.append(current_dir)
import glob
import numpy as np
from PIL import Image
from Experiment.estimate import compute_ncc, compute_dice_max_k
from register.non_linear_align import FlowAligner, SEAMLeSS
from serial_reg import fixed_first_frame, fixed_first_with_last_as_virtual_section, weight_assignment, vvote
from SIFTFlow_serial import SFTFlow_register


# method
def serial_estimate(path_data_lst, aligner):
    """
    input:
    """
    field_lst = []
    img_r = cv2.imread(path_data_lst[0], cv2.IMREAD_GRAYSCALE)
    for i in range(1, len(path_data_lst)):
        img_m = cv2.imread(path_data_lst[i], cv2.IMREAD_GRAYSCALE)
        field = aligner.generate_field(img_r, img_m)
        img_r = aligner.warp_with_field(img_m, field.clone())
        field_lst.append(aligner.generate_field(img_r, img_m))
    return field_lst


def serial_register(dir_input, dir_output, aligner, axis_method=None):
    # <editor-fold desc="find wanted image in dirs, and make the output dir">
    path_d_lst = glob.glob(os.path.join(dir_input, "serial32", "data", "*.png"))
    path_l_lst = glob.glob(os.path.join(dir_input, "serial32", "label", "*.png"))
    path_d_lst.sort()
    path_l_lst.sort()
    os.makedirs(os.path.join(dir_output, axis_method, "data_finished"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, axis_method, "label_finished"), exist_ok=True)
    img_lst = []
    label_lst = []
    # </editor-fold>

    # <editor-fold desc="read image as lists: img_lst and label_lst (optional)">
    for i in range(len(path_d_lst)):
        if i == 0 or i == len(path_d_lst) - 1:
            img_lst.append(cv2.imread(path_d_lst[i], cv2.IMREAD_GRAYSCALE))
        else:
            img_lst.append(cv2.imread(path_d_lst[i], cv2.IMREAD_GRAYSCALE))
        if path_l_lst:
            if i == 0 or i == len(path_d_lst) - 1:
                label_lst.append(cv2.imread(path_l_lst[i], cv2.IMREAD_UNCHANGED))
            else:
                label_lst.append(cv2.imread(path_l_lst[i], cv2.IMREAD_UNCHANGED))
    # </editor-fold>

    # <editor-fold desc="serial align, direct, structure regression or vvote, output is img_w_lst and img_l_lst(optional)">
    if axis_method == "sr":
        serial_aligner = fixed_first_with_last_as_virtual_section
    elif axis_method == "vvote":
        serial_aligner = vvote
    else:
        serial_aligner = fixed_first_frame
    if path_l_lst:
        img_w_lst, img_l_lst = serial_aligner(img_lst, aligner, label_lst)
        img_w_lst += [cv2.imread(path_d_lst[-1], cv2.IMREAD_GRAYSCALE)]
        img_l_lst += [cv2.imread(path_l_lst[-1], cv2.IMREAD_UNCHANGED)]
    else:
        img_w_lst, _ = serial_aligner(img_lst, aligner)
        img_w_lst += [cv2.imread(path_d_lst[-1], cv2.IMREAD_GRAYSCALE)]
    # </editor-fold>

    # <editor-fold desc="write the finished image"
    for i in range(len(img_w_lst)):
        cv2.imwrite(path_d_lst[i].replace(dir_input + "serial32/data",
                                          os.path.join(dir_output, axis_method, "data_finished")), img_w_lst[i])
        if path_l_lst:
            cv2.imwrite(path_d_lst[i].replace(dir_input + "serial32/data",
                                              os.path.join(dir_output, axis_method, "label_finished")), img_l_lst[i])
    # </editor-fold>


# Ext
def ARFlow_serial(dir_input, output):
    model_name = "pwclite_ar.tar"
    aligner = FlowAligner('../models/{}'.format(model_name), True, "cuda")
    serial_register(dir_input, os.path.join(output, "ARFlow"), aligner, 'serial')
    serial_register(dir_input, os.path.join(output, "ARFlow"), aligner, 'vvote')
    serial_register(dir_input, os.path.join(output, "ARFlow"), aligner, 'sr')


def ARFlow_ft_serial(dir_input, output):
    model_name = "ARFlow_ft_c_9.pth"
    aligner = FlowAligner('../models/{}'.format(model_name), True, "cuda")
    serial_register(dir_input, os.path.join(output, "ARFlow_ft"), aligner, 'serial')
    serial_register(dir_input, os.path.join(output, "ARFlow_ft"), aligner, 'vvote')
    serial_register(dir_input, os.path.join(output, "ARFlow_ft"), aligner, 'sr')


def FESF_serial(dir_input, output):
    model_name = "FE_serial.pth"
    aligner = FlowAligner('../models/{}'.format(model_name), True, "cuda")
    serial_register(dir_input, os.path.join(output, "FESF"), aligner, 'serial')
    serial_register(dir_input, os.path.join(output, "FESF"), aligner, 'vvote')
    serial_register(dir_input, os.path.join(output, "FESF"), aligner, 'sr')


def SEAMLeSS_serial(dir_input, output):
    aligner = SEAMLeSS()
    serial_register(dir_input, os.path.join(output, "SEAMLeSS"), aligner, 'serial')
    serial_register(dir_input, os.path.join(output, "SEAMLeSS"), aligner, 'vvote')
    serial_register(dir_input, os.path.join(output, "SEAMLeSS"), aligner, 'sr')


def SIFTFlow_serial(dir_input, output):
    model_name = "pwclite_ar.tar"
    aligner = FlowAligner('../models/{}'.format(model_name), True, "cuda")
    SFTFlow_register(dir_input, os.path.join(output, "SIFTFlow"), aligner, 'serial')
    SFTFlow_register(dir_input, os.path.join(output, "SIFTFlow"), aligner, 'vvote')
    SFTFlow_register(dir_input, os.path.join(output, "SIFTFlow"), aligner, 'sr')


#

# compute the metric, including ncc and dice50
def compute_metric_with_name(dir_input, dir_rst, name):
    print("serial result:")
    compute_metric(dir_input, dir_rst, name, 'serial')
    print("vvote result:")
    compute_metric(dir_input, dir_rst, name, 'vvote')
    print("structure regression result:")
    compute_metric(dir_input, dir_rst, name, 'sr')


def compute_metric(dir_input, dir_rst, name, method):
    path_groundtruth = glob.glob(os.path.join(dir_input, "serial32", "data_groundtruth", "*.png"))
    path_data_finished = glob.glob(os.path.join(dir_rst, name, method, "data_finished", "*.png"))
    path_label_groundtruth = glob.glob(os.path.join(dir_input, "serial32", "label_groundtruth", "*.png"))
    path_label_finished = glob.glob(os.path.join(dir_rst, name, method, "label_finished", "*.png"))
    path_groundtruth.sort()
    path_data_finished.sort()
    path_label_groundtruth.sort()
    path_label_finished.sort()
    ncc_gt_lst = []
    dice50_lst = []
    for i in range(len(path_groundtruth) - 1):
        path_gt = path_groundtruth[i]
        path_f = path_data_finished[i]
        img_gt = np.array(Image.open(path_gt))
        img_f = np.array(Image.open(path_f))
        ncc_gt_lst.append(compute_ncc(img_f, img_gt))
        if path_label_groundtruth:
            path_l_gt = path_label_groundtruth[i]
            path_l = path_label_finished[i]
            img_l = cv2.imread(path_l, cv2.IMREAD_UNCHANGED)
            img_l_gt = cv2.imread(path_l_gt, cv2.IMREAD_UNCHANGED)
            dice50_lst.append(compute_dice_max_k(img_l, img_l_gt, 50))
    print("ncc with ground truth:{:.3f}+-{:.3f}".format(np.mean(ncc_gt_lst), np.std(ncc_gt_lst)))
    if dice50_lst:
        print("dice with ground truth:{:.3f}+-{:.3f}".format(np.mean(dice50_lst), np.std(dice50_lst)))


if __name__ == "__main__":
    DATASET_NAME = "FIB_mito"
    # DATASET_NAME = "CREMIA"
    input = "/home/xint/mnt/DATA/FESFData/{}/".format(DATASET_NAME)
    output = "Rst/{}/serial32".format(DATASET_NAME)
    # SIFTFlow_serial(input, output)
    compute_metric_with_name(input, output, 'SIFTFlow')
    print("SIFTFlow finished")
    # SEAMLeSS_serial(input, output)
    compute_metric_with_name(input, output, 'SEAMLeSS')
    print("SEAMLeSS finished")
    # ARFlow_serial(input, output)
    compute_metric_with_name(input, output, 'ARFlow')
    print("ARFlow finished")
    # ARFlow_ft_serial(input, output)
    compute_metric_with_name(input, output, 'ARFlow_ft')
    print("ARFlow_ft finished")
    # FESF_serial(input, output)
    compute_metric_with_name(input, output, 'FESF')
    print("FESF finished")
    compute_metric_with_name(input, output, 'Elastic')

