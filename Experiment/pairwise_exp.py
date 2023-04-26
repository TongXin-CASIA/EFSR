import os
import sys
# Add the directory containing the Experiment package to the PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(os.path.join(__file__,'..')))
sys.path.append(current_dir)
import cv2
import glob
import numpy as np
from PIL import Image
import skimage
from Experiment.estimate import compute_ncc, compute_dice_max_k
from register.non_linear_align import FlowAligner, SEAMLeSS


def pairwise_align(img_r, img_m, aligner, label=None):
    field = aligner.generate_field(img_r, img_m)
    img_w = aligner.warp_with_field(img_m, field.clone())
    if label is not None:
        label = aligner.warp_with_field(label, field.clone(), "nearest")
        return img_w, label
    else:
        return img_w


def batch_pairwise(dir_input, dir_output, aligner):
    path_r_lst = glob.glob(os.path.join(dir_input, "pairwise", "data_reference", "*.png"))
    path_m_lst = glob.glob(os.path.join(dir_input, "pairwise", "data_moving", "*.png"))
    path_l_lst = glob.glob(os.path.join(dir_input, "pairwise", "label_moving", "*.png"))
    path_r_lst.sort()
    path_m_lst.sort()
    path_l_lst.sort()
    os.makedirs(os.path.join(dir_output, "data_finished"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "label_finished"), exist_ok=True)
    for i in range(len(path_r_lst)):
        path_r = path_r_lst[i]
        path_m = path_m_lst[i]
        img_r = np.array(Image.open(path_r))
        img_m = np.array(Image.open(path_m))
        if path_l_lst:
            path_l = path_l_lst[i]
            img_l = cv2.imread(path_l, cv2.IMREAD_UNCHANGED)
            img_w, img_l = pairwise_align(img_r, img_m, aligner, img_l)
            cv2.imwrite(
                path_r.replace(dir_input + "pairwise/data_reference", os.path.join(dir_output, "label_finished")),
                img_l)
        else:
            img_w = pairwise_align(img_r, img_m, aligner)
        cv2.imwrite(path_r.replace(dir_input + "pairwise/data_reference", os.path.join(dir_output, "data_finished")),
                    img_w)


# experiment and generate the result
def ARFlow_pairwise(dir_input, output):
    model_name = "pwclite_ar.tar"
    aligner = FlowAligner('../models/{}'.format(model_name), True, "cuda")
    batch_pairwise(dir_input, os.path.join(output, "ARFlow"), aligner)


def ARFlow_ft_pairwise(dir_input, output):
    model_name = "ARFlow_ft_c_9.pth"
    aligner = FlowAligner('../models/{}'.format(model_name), True, "cuda")
    batch_pairwise(dir_input, os.path.join(output, "ARFlow_ft"), aligner)


def FESF_pairwise(dir_input, output):
    model_name = "FE_pairwise.pth"
    aligner = FlowAligner('../models/{}'.format(model_name), True, "cuda")
    batch_pairwise(dir_input, os.path.join(output, "FESF"), aligner)


def SEAMLeSS_pairwise(dir_input, output):
    aligner = SEAMLeSS()
    batch_pairwise(dir_input, os.path.join(output, "SEAMLeSS"), aligner)


# compute the metric
def compute_metric(dir_input, dir_rst, name):
    path_reference = glob.glob(os.path.join(dir_input, "pairwise", "data_reference", "*.png"))
    path_groundtruth = glob.glob(os.path.join(dir_input, "pairwise", "data_groundtruth", "*.png"))
    path_data_finished = glob.glob(os.path.join(dir_rst, name, "data_finished", "*.png")) +\
                         glob.glob(os.path.join(dir_rst, name, "data_finished", "*.tif"))
    path_label_groundtruth = glob.glob(os.path.join(dir_input, "pairwise", "label_groundtruth", "*.png"))
    path_label_finished = glob.glob(os.path.join(dir_rst, name, "label_finished", "*.png")) + \
                          glob.glob(os.path.join(dir_rst, name, "label_finished", "*.tif"))
    path_reference.sort()
    path_groundtruth.sort()
    path_data_finished.sort()
    path_label_groundtruth.sort()
    path_label_finished.sort()
    ncc_reference_lst = []
    ncc_gt_lst = []
    dice50_lst = []
    for i in range(len(path_reference)):
        path_r = path_reference[i]
        path_gt = path_groundtruth[i]
        path_f = path_data_finished[i]
        img_r = np.array(Image.open(path_r))
        img_gt = np.array(Image.open(path_gt))
        img_f = np.array(Image.open(path_f))
        if path_f[-3:] == 'tif':
            img_f = skimage.io.imread(path_f)
            img1 = img_f[0]
            row = img1.sum(0)
            col = img1.sum(1)
            x = np.argwhere(row != 0)[0][0]
            y = np.argwhere(col != 0)[0][0]
            img_f = img_f[1][y:y+1024, x:x+1024]
        ncc_reference_lst.append(compute_ncc(img_r, img_f))
        ncc_gt_lst.append(compute_ncc(img_gt, img_f))
        if path_label_groundtruth:
            path_l_gt = path_label_groundtruth[i]
            path_l = path_label_finished[i]
            img_l = cv2.imread(path_l, cv2.IMREAD_UNCHANGED)
            img_l_gt = cv2.imread(path_l_gt, cv2.IMREAD_UNCHANGED)
            if path_f[-3:] == 'tif':
                img_l = img_l[y:y + 1024, x:x + 1024]
            dice50_lst.append(compute_dice_max_k(img_l_gt, img_l, 50))
    print("ncc with reference:{:.3f}+-{:.3f}".format(np.mean(ncc_reference_lst), np.std(ncc_reference_lst)))
    print("ncc with ground truth:{:.3f}+-{:.3f}".format(np.mean(ncc_gt_lst), np.std(ncc_gt_lst)))
    print("dice with ground truth:{:.3f}+-{:.3f}".format(np.mean(dice50_lst), np.std(dice50_lst)))


if __name__ == "__main__":
    DATASET_NAME = "FIB_mito"
    # DATASET_NAME = "CREMIA"
    input = "/home/xint/mnt/DATA/FESFData/{}/".format(DATASET_NAME)
    output = "Rst/{}/pairwise".format(DATASET_NAME)
    # # ARFlow_pairwise(input, output)
    # compute_metric(input, output, 'ARFlow')
    # print("ARFlow finished")
    # # ARFlow_ft_pairwise(input, output)
    # compute_metric(input, output, 'ARFlow_ft')
    # print("ARFlow_ft finished")
    # FESF_pairwise(input, output)
    # compute_metric(input, output, 'FESF')
    # print("FESF finished")
    # # SEAMLeSS_pairwise(input, output)
    # compute_metric(input, output, 'SEAMLeSS')
    # print("SEAMLeSS finished")
    # compute_metric(input, output, 'SIFTFlow')
    # print("SIFTFlow finished")
    # compute_metric(input, output, 'Elastic')
    compute_metric(input, output, 'Flowformer')