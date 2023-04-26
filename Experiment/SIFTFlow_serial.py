import os
import cv2
import glob
import torch
from scipy.io import loadmat
from Experiment.serial_reg import weight_assignment


def load_field(path):
    mat = loadmat(path)
    vx, vy = torch.from_numpy(mat['vx']).unsqueeze(0), torch.from_numpy(mat['vy']).unsqueeze(0)
    field = torch.concat([vx, vy]).unsqueeze(0).float()
    return field


def fixed_first_with_last_as_virtual_section_load_field(img_lst, dir_output, aligner,  lb_lst=None):
    img_w_lst, field_lst = fixed_first_frame_load_field(img_lst, dir_output, aligner)
    img_last = img_w_lst[-1]
    img_r = img_lst[-1]
    flow_last = load_field(os.path.join(dir_output,"field", "error.mat"))
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


def fixed_first_frame_load_field(img_lst, dir_output, aligner,  lb_lst=None):
    """
    params: img_lst
    """
    img_w_lst = []
    path_f_lst = glob.glob(os.path.join(dir_output, "field", "*_s.mat"))
    path_f_lst.sort()
    field_lst = [None] + [load_field(path_f) for path_f in path_f_lst]
    for i in range(len(img_lst) - 1):
        if i == 0:
            img_w = img_lst[i]
        else:
            img_m = img_lst[i]
            field = field_lst[i]
            img_w = aligner.warp_with_field(img_m, field.clone())
            if lb_lst is not None:
                lb_lst[i] = aligner.warp_with_field(lb_lst[i], field.clone(), 'nearest')
        img_w_lst.append(img_w)
    if lb_lst is not None:
        return img_w_lst, lb_lst
    return img_w_lst, field_lst


def vvote_load_field(img_lst, dir_output, aligner,  lb_lst=None):
    """
    params: img_lst
    """
    img_w_lst = []
    path_f_lst = glob.glob(os.path.join(dir_output, "field", "*_v.mat"))
    path_f_lst.sort()
    field_lst = [None] + [load_field(path_f) for path_f in path_f_lst]
    for i in range(len(img_lst) - 1):
        if i == 0:
            img_w = img_lst[i]
        else:
            img_m = img_lst[i]
            field = field_lst[i]
            img_w = aligner.warp_with_field(img_m, field.clone())
            if lb_lst is not None:
                lb_lst[i] = aligner.warp_with_field(lb_lst[i], field.clone(), 'nearest')
        img_w_lst.append(img_w)
    if lb_lst is not None:
        return img_w_lst, lb_lst
    return img_w_lst, field_lst


def SFTFlow_register(dir_input, dir_output, aligner, axis_method=None):
    # <editor-fold desc="sifflow register">
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
        serial_aligner = fixed_first_with_last_as_virtual_section_load_field
    elif axis_method == "vvote":
        serial_aligner = vvote_load_field
    else:
        serial_aligner = fixed_first_frame_load_field
    if path_l_lst:
        img_w_lst, img_l_lst = serial_aligner(img_lst, dir_output, aligner, label_lst)
        img_w_lst += [cv2.imread(path_d_lst[-1], cv2.IMREAD_GRAYSCALE)]
        img_l_lst += [cv2.imread(path_l_lst[-1], cv2.IMREAD_UNCHANGED)]
    else:
        img_w_lst, _ = serial_aligner(img_lst, dir_output, aligner)
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
    # </editor-fold>
