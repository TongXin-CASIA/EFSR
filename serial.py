import os
import cv2
import glob
import shutil
import numpy as np
import torch
import argparse
from Architecture.UTR import UTRNet
from Experiment.estimate import compute_ncc, compute_dice_max_k, compute_dice_coefficients
from register.non_linear_align import FlowAligner, img2tensor
from register.flow import warp_image_two_steps


@torch.no_grad()
def weight_assignment(img_lst, mode="linear"):
    weight_lst = []
    shape = img_lst[0].shape
    if mode == "linear":
        weight_lst.append(0)
        for i in range(1, len(img_lst) - 1):
            weight_lst.append((i - 1) / (len(img_lst) - 3))
    else:
        # compute des
        model = UTRNet(os.path.dirname(__file__) + '/../Architecture/AngleScale.pth')
        des_lst = [model(img2tensor(img)) for img in img_lst]
        diff_torch = torch.cat([(des_lst[i] - des_lst[i - 1]).pow(2).sum(1).sqrt().detach()
                                for i in range(1, len(img_lst) - 1)])

        # compute weight
        if mode == "global":
            diff_torch = diff_torch.mean([1, 2])
            diff_sum = diff_torch.sum(0)
            diff_acc = 0
            for d in diff_torch:
                diff_acc += d
                weight_lst.append(diff_acc / diff_sum)
        elif mode == "local":
            diff_sum = diff_torch.sum(0)
            diff_acc = 0
            for d in diff_torch:
                diff_acc += d
                weight = torch.nn.functional.interpolate((diff_acc / diff_sum).unsqueeze(0).unsqueeze(0), shape)
                weight_lst.append(weight)
        else:
            raise NotImplementedError("Unsupported mode")
    return weight_lst


def fixed_first_frame(img_lst, aligner, lb_lst=None):
    """
    params: img_lst
    """
    img_w_lst = []
    field_lst = []
    for i in range(len(img_lst) - 1):
        if i == 0:
            img_w = img_lst[i]
            field = None
        else:
            img_r = img_w
            img_m = img_lst[i]
            field = aligner.generate_field(img_r, img_m)
            img_w = aligner.warp_with_field(img_m, field.clone())
            if lb_lst is not None:
                lb_lst[i] = aligner.warp_with_field(lb_lst[i], field.clone(), 'nearest')
        field_lst.append(field)
        img_w_lst.append(img_w)
    if lb_lst is not None:
        return img_w_lst, lb_lst
    return img_w_lst, field_lst


def fixed_first_with_last_as_virtual_section(img_lst, aligner, lb_lst=None):
    img_w_lst, field_lst = fixed_first_frame(img_lst, aligner)
    img_last = img_w_lst[-1]
    img_r = img_lst[-1]
    flow_last = aligner.generate_field(img_r, img_last)
    weight_lst = weight_assignment(img_w_lst + [img_r], 'global')
    # warp all the sections via the linear flow
    for i in range(1, len(img_w_lst)):
        alpha = weight_lst[i - 1]
        # warp field
        field = field_lst[i]
        img_w_lst[i] = aligner.warp_with_field_two_steps(img_lst[i], flow_last.clone() * alpha, field.clone())
        if lb_lst is not None:
            lb_lst[i] = aligner.warp_with_field_two_steps(lb_lst[i], flow_last.clone() * alpha, field.clone(),
                                                          'nearest')
    if lb_lst is not None:
        return img_w_lst, lb_lst
    return img_w_lst, None


def vvote(img_lst, aligner, lb_lst=None):
    n = 3
    img_w_lst = [img_lst[0]]
    for i in range(1, min(n, len(img_lst))):
        img_r = img_w_lst[i - 1]
        img_m = img_lst[i]
        field = aligner.generate_field(img_r, img_m)
        img_w = aligner.warp_with_field(img_r, field.clone())
        img_w_lst.append(img_w)
    for i in range(n, len(img_lst) - 1):
        field = []
        for j in range(n):
            img_r = img_w_lst[i - 1 - j]
            img_m = img_lst[i]
            field.append(aligner.generate_field(img_r, img_m))
        field = torch.median(torch.cat(field), dim=0, keepdims=True)[0]
        img_w = aligner.warp_with_field(img_lst[i], field.clone())
        img_w_lst.append(img_w)
        if lb_lst is not None:
            lb_lst[i] = aligner.warp_with_field(lb_lst[i], field.clone(), 'nearest')
    if lb_lst is not None:
        return img_w_lst, lb_lst
    return img_w_lst, None


# create data
def create_data(input_origin_dir, input_warped_dir, output_gt_dir, output_dataset_dir):
    """
    :param input_origin_dir:
    :param input_warped_dir:
    :param output_gt_dir:
    :param output_dataset_dir:
    :return:
    """
    crop_size = 1024
    path_o = glob.glob(input_origin_dir + "*.png")
    path_w = glob.glob(input_warped_dir + "*.png")
    path_o.sort()
    path_w.sort()
    path_gt = path_o
    path_d = [path_o[0]] + path_w[1:-1] + [path_o[-1]]
    # create ouput_gt_dir
    if not os.path.exists(output_gt_dir):
        os.makedirs(output_gt_dir)
    # create ouput_dataset_dir
    if not os.path.exists(output_dataset_dir):
        os.makedirs(output_dataset_dir)
    for data, gt in zip(path_d, path_gt):
        img_data = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
        img_gt = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
        crop = (img_data.shape[0] - crop_size) // 2
        img_data = img_data[crop:-crop, crop:-crop]
        img_gt = img_gt[crop:-crop, crop:-crop]
        cv2.imwrite(output_gt_dir + "/" + gt.split("/")[-1], img_gt)
        cv2.imwrite(output_dataset_dir + "/" + data.split("/")[-1], img_data)


def serial_reg(dir_serial, dir_out, aligner, structure_regression="error"):
    path_s_lst = glob.glob(dir_serial + "*serial.png")
    path_s_lst.sort()
    img_s_lst = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in path_s_lst]
    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
    os.makedirs(dir_out)
    if structure_regression == "error":
        img_w_lst = fixed_first_with_last_as_virtual_section(img_s_lst, aligner)
    elif structure_regression == "vvote":
        img_w_lst = vvote(img_s_lst, aligner)
    else:
        img_w_lst = fixed_first_frame(img_s_lst, aligner)
    for i, img_w in enumerate(img_w_lst):
        cv2.imwrite(os.path.join(dir_out, "{:03d}.png".format(i)), img_w)
    cv2.imwrite(os.path.join(dir_out, "{:03d}.png".format(i+1)), img_s_lst[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--mode", type=str, default="error")
    parser.add_argument("--model_path", type=str, default='../models/FE.pth')
    parser.add_argument("--output_dir", type=str, default='output')
    args = parser.parse_args()
    aligner = FlowAligner(args.model_path, True, "cuda")
    serial_reg(args.input_dir, args.output_dir, aligner, args.mode)
