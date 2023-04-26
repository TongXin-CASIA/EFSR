"""
serial align experiment:
input dir: stack (the first and the end is the ground truth)
output dir: stack
"""

import os
import cv2
import glob
import shutil
import numpy as np
import torch
from Architecture.UTR import UTRNet
from Experiment.estimate import compute_ncc, compute_dice_max_k, compute_dice_coefficients
from register.non_linear_align import FlowAligner, img2tensor, SEAMLeSS
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


def serial_reg(dir_serial, dir_gt, dir_out, aligner, structure_regression="error"):
    path_s_lst = glob.glob(dir_serial + "*serial.png")
    path_gt_lst = glob.glob(dir_gt + "*.png")
    path_l_lst = glob.glob(dir_serial + "*label.png")
    path_lo_lst = glob.glob(dir_serial + "*label_org.png")
    path_s_lst.sort()
    path_gt_lst.sort()
    path_l_lst.sort()
    path_lo_lst.sort()
    img_s_lst = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in path_s_lst]
    img_gt_lst = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in path_gt_lst]
    img_l_lst = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in path_l_lst]
    img_lo_lst = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in path_lo_lst]
    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
    os.makedirs(dir_out)
    if structure_regression == "error":
        img_w_lst, img_l_lst = fixed_first_with_last_as_virtual_section(img_s_lst, aligner, img_l_lst)
    elif structure_regression == "vvote":
        img_w_lst, img_l_lst = vvote(img_s_lst, aligner, img_l_lst)
    else:
        img_w_lst, img_l_lst = fixed_first_frame(img_s_lst, aligner, img_l_lst)
    ncc = []
    dice = []
    dice50 = []
    for img_w, img_gt, img_l, img_lo, path in zip(img_w_lst, img_gt_lst, img_l_lst, img_lo_lst, path_s_lst):
        ncc.append(compute_ncc(img_w, img_gt))
        dice.append(compute_dice_coefficients(img_l, img_lo))
        dice50.append(compute_dice_max_k(img_l, img_lo, 50))
        cv2.imwrite(dir_out + "/" + path.split("/")[-1], img_w)
    print("ncc: ", np.mean(ncc))
    print("dice: ", np.mean(dice))
    print("dice50: ", np.mean(dice50))
    return img_w_lst + [img_s_lst[-1]], img_l_lst


def ext_call_serial(name, model_path, device):
    input_dir = "/home/xint/mnt/DATA/opticalData/CREMI_A_Serial32/"
    input_gt_dir = "/home/xint/mnt/DATA/opticalData/CREMI_A_O/"
    path = os.path.dirname(os.path.dirname(__file__))
    output_dir = path + '/rst/{}_s'.format(name)
    is_c = True if name[-1] == 'c' else False
    aligner = FlowAligner(model_path, is_c, device)
    return serial_reg(input_dir, input_gt_dir, output_dir, aligner, "error")


def convert_sight(img_lst, thickness=1, axis_num=10, axis_type='z', label_lst=None):
    img_stack = np.array(img_lst)
    label_stack = np.array(label_lst)
    if axis_type == 'x':
        img_stack = img_stack[:, :, axis_num]
        axis1_scale, axis2_scale = 1, thickness
        if label_stack is not None:
            label_stack = label_stack[:, :, axis_num]
    elif axis_type == 'y':
        img_stack = img_stack[:, axis_num, :]
        axis1_scale, axis2_scale = 1, thickness
        if label_stack is not None:
            label_stack = label_stack[:, axis_num, :]
    elif axis_type == 'x':
        img_stack = img_stack[axis_num, :, :]
        axis1_scale, axis2_scale = 1, 1
        if label_stack is not None:
            label_stack = label_stack[axis_num, :, :]
    else:
        raise ValueError("Unknown axis")
    img_stack = cv2.resize(img_stack, None, fx=axis1_scale, fy=axis2_scale, interpolation=cv2.INTER_NEAREST)
    if label_lst is not None:
        np.random.seed(0)
        color_lst = np.random.randint(0, 255, [256, 3])
        color_lst[0] = np.array([0, 0, 0])
        label_stack = cv2.resize(label_stack, None, fx=axis1_scale, fy=axis2_scale, interpolation=cv2.INTER_NEAREST)
        label_colored = color_lst[label_stack.astype(np.uint8)]
        img_stack = cv2.cvtColor(img_stack, cv2.COLOR_GRAY2RGB)
        img_stack = 0.7 * img_stack + 0.3 * label_colored
    return img_stack


def draw_module_rst():
    exp_lst = {'ARFlow_c': 'pwclite_ar.tar',
               'ARFlow_ft_c': 'ARFlow_ft_c_9.pth',
               'FE_ft_c': 'FE_serial.pth'}
    dir_input = "/home/xint/mnt/DATA/opticalData/CREMI_A_Serial32/"
    dir_gt = "/home/xint/mnt/DATA/opticalData/CREMI_A_O/"
    for exp_name in exp_lst:
        print("Experiment %s" % exp_name)
        path_model = '../models/{}'.format(exp_lst[exp_name])
        dir_output = '../rst/{}_s'.format(exp_name)
        is_c = True if exp_name[-1] == 'c' else False
        aligner = FlowAligner(path_model, is_c, "cuda")
        img_w_lst, img_l_lst = serial_reg(dir_input, dir_gt, dir_output, aligner, "error")
        longisection = convert_sight(img_w_lst, 10, 120, 'y', img_l_lst)
        cv2.imwrite(dir_output + '_error.png', longisection)
        img_w_lst, img_l_lst = serial_reg(dir_input, dir_gt, dir_output, aligner, "none")
        longisection = convert_sight(img_w_lst, 10, 120, 'y', img_l_lst)
        cv2.imwrite(dir_output + '_none.png', longisection)
        img_w_lst, img_l_lst = serial_reg(dir_input, dir_gt, dir_output, aligner, "vvote")
        longisection = convert_sight(img_w_lst, 10, 120, 'y', img_l_lst)
        cv2.imwrite(dir_output + '_vvote.png', longisection)
    exp_name = "SEAMLeSS"
    print("Experiment %s" % exp_name)
    dir_output = '../rst/{}_s'.format(exp_name)
    aligner = SEAMLeSS()
    img_w_lst, img_l_lst = serial_reg(dir_input, dir_gt, dir_output, aligner, "error")
    longisection = convert_sight(img_w_lst, 10, 120, 'y', img_l_lst)
    cv2.imwrite(dir_output + '_error.png', longisection)
    img_w_lst, img_l_lst = serial_reg(dir_input, dir_gt, dir_output, aligner, "none")
    longisection = convert_sight(img_w_lst, 10, 120, 'y', img_l_lst)
    cv2.imwrite(dir_output + '_none.png', longisection)
    img_w_lst, img_l_lst = serial_reg(dir_input, dir_gt, dir_output, aligner, "vvote")
    longisection = convert_sight(img_w_lst, 10, 120, 'y', img_l_lst)
    cv2.imwrite(dir_output + '_vvote.png', longisection)


def draw_org_label():
    dir_input = "/home/xint/mnt/DATA/opticalData/CREMI_A_Serial125/"
    dir_gt = "/home/xint/mnt/DATA/opticalData/CREMI_A_O/"
    path_s_lst = glob.glob(dir_input + "*serial.png")
    path_gt_lst = glob.glob(dir_gt + "*.png")
    path_l_lst = glob.glob(dir_input + "*label.png")
    path_lo_lst = glob.glob(dir_input + "*label_org.png")
    path_s_lst.sort()
    path_gt_lst.sort()
    path_l_lst.sort()
    path_lo_lst.sort()
    img_s_lst = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in path_s_lst]
    img_gt_lst = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in path_gt_lst]
    img_l_lst = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in path_l_lst]
    img_lo_lst = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in path_lo_lst]
    longisection = convert_sight(img_s_lst, 10, 120, 'y', img_l_lst)
    cv2.imwrite('../rst/org125.png', longisection)
    longisection = convert_sight(img_gt_lst, 10, 120, 'y', img_lo_lst)
    cv2.imwrite('../rst/gt125.png', longisection)


def show_dir_rst(dir_input):
    dir_gt = "/home/xint/mnt/DATA/opticalData/CREMI_A_O/"
    path_s_lst = glob.glob(dir_input + "*serial.png")
    path_gt_lst = glob.glob(dir_gt + "*.png")
    path_l_lst = glob.glob(dir_input + "*label.png")
    path_lo_lst = glob.glob(dir_input + "*label_org.png")
    path_s_lst.sort()
    path_gt_lst.sort()
    path_l_lst.sort()
    path_lo_lst.sort()
    img_s_lst = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in path_s_lst]
    img_gt_lst = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in path_gt_lst][:32]
    img_l_lst = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in path_l_lst]
    img_lo_lst = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in path_lo_lst]
    ncc = []
    dice = []
    dice50 = []
    for img_s, img_gt, img_l, img_lo in zip(img_s_lst, img_gt_lst, img_l_lst, img_lo_lst):
        ncc.append(compute_ncc(img_s, img_gt))
        dice.append(compute_dice_coefficients(img_l, img_lo))
        dice50.append(compute_dice_max_k(img_l, img_lo, 50))
    print("ncc: ", np.mean(ncc))
    print("dice: ", np.mean(dice))
    print("dice50: ", np.mean(dice50))
    longisection = convert_sight(img_s_lst, 10, 120, 'y', img_l_lst)
    cv2.imwrite('../rst/elastic_s.png', longisection)


def total_exp():
    from copy import deepcopy
    mode_rs = "none"
    exp_lst = {'ARFlow_c': 'pwclite_ar.tar',
               'ARFlow_ft_c': 'ARFlow_ft_c_9.pth',
               'FE_ft_c': 'FE_serial.pth'}
    dir_input = "/home/xint/mnt/DATA/opticalData/CREMI_A_Serial125/"
    dir_gt = "/home/xint/mnt/DATA/opticalData/CREMI_A_O/"

    # for exp_name in exp_lst:
    #     print("Experiment %s" % exp_name)
    #     path_model = '../models/{}'.format(exp_lst[exp_name])
    #     dir_output = '../rst/{}_t'.format(exp_name)
    #     is_c = True if exp_name[-1] == 'c' else False
    #     aligner = FlowAligner(path_model, is_c, "cuda")
    #     img_total_lst = []
    #     label_total_lst = []
    #     for i in range(5):
    #         img_w_lst, img_l_lst = serial_reg(dir_input + str(i) + '/', dir_gt + str(i) + '/', dir_output, aligner,
    #                                           mode_rs)
    #         img_total_lst.extend(img_w_lst)
    #         label_total_lst.extend(img_l_lst)
    #     longisection = convert_sight(img_total_lst, 10, 120, 'y', label_total_lst)
    # cv2.imwrite(dir_output + '_none.png', longisection)

    exp_name = "SEAMLeSS"
    print("Experiment %s" % exp_name)
    dir_output = '../rst/{}_t'.format(exp_name)
    aligner = SEAMLeSS()
    img_total_lst = []
    label_total_lst = []
    for i in range(5):
        img_w_lst, img_l_lst = serial_reg(dir_input + str(i) + '/', dir_gt + str(i) + '/', dir_output, aligner,
                                          'vvote')
        img_total_lst.extend(img_w_lst)
        label_total_lst.extend(img_l_lst)
    longisection = convert_sight(img_total_lst, 10, 120, 'y', label_total_lst)
    cv2.imwrite(dir_output + '_vvote.png', longisection)
    print('finished')


def total_elastic():
    import skimage.io as io
    dir_input = "/media/ExtHDD01/xint/PyProjects/FESF/rst/Elastic_t/"
    dir_gt = "/home/xint/mnt/DATA/opticalData/CREMI_A_O/"

    exp_name = "Elastic"
    print("Experiment %s" % exp_name)
    dir_output = '../rst/{}_t'.format(exp_name)
    img_total_lst = []
    label_total_lst = []
    for i in range(5):
        img_w = io.imread(dir_input + "Serial%d.tif" % (i))
        img_l = io.imread(dir_input + "Labels%d.tif" % (i))
        img = img_w[0]
        l, t = img.sum(0).nonzero()[0][0], img.sum(1).nonzero()[0][0]
        img_w = img_w[:, t:t + 1024, l:l + 1024]
        img = img_l[0]
        l, t = img.sum(0).nonzero()[0][0], img.sum(1).nonzero()[0][0]
        img_l = img_l[:, t:t + 1024, l:l + 1024]
        img_w_lst = [img for img in img_w]
        img_l_lst = [img for img in img_l]
        img_total_lst.extend(img_w_lst)
        label_total_lst.extend(img_l_lst)
    longisection = convert_sight(img_total_lst, 10, 120, 'y', label_total_lst)
    cv2.imwrite(dir_output + '.png', longisection)
    print('finished')


def main():
    model_lst = {'ARFlow_c': 'pwclite_ar.tar',
                 'ARFlow_ft_c': 'ARFlow_ft_c_9.pth',
                 'FE_ft_c': 'FE_serial.pth'}
    input_dir = "/home/xint/mnt/DATA/opticalData/CREMI_A_Serial32/"
    input_gt_dir = "/home/xint/mnt/DATA/opticalData/CREMI_A_O/"
    mode = "error"
    for name in model_lst:
        print("Experiment %s" % name)
        model_path = '../models/{}'.format(model_lst[name])
        output_dir = '../rst/{}_s'.format(name)
        is_c = True if name[-1] == 'c' else False
        aligner = FlowAligner(model_path, is_c, "cuda")
        serial_reg(input_dir, input_gt_dir, output_dir, aligner, mode)
    name = 'SEAMLeSS'
    aligner = SEAMLeSS()
    output_dir = '../rst/{}_s'.format(name)
    serial_reg(input_dir, input_gt_dir, output_dir, aligner, mode)


if __name__ == "__main__":
    main()
    # draw_module_rst()
    # draw_org_label()
    # total_exp()
    # show_dir_rst("/media/ExtHDD01/xint/PyProjects/FESF/rst/Elastic_s/")
    # total_elastic()
