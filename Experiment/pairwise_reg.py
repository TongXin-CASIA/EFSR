"""
pairwise experiment
input: two images(img_r, img_m)
output: three images(img_r, img_w, img_r)
"""
import os
import cv2
import glob
import shutil
import argparse
import numpy as np
from PIL import Image
from Experiment.estimate import compute_ncc, compute_dice_coefficients, compute_dice_max_k
from register.non_linear_align import FlowAligner, SEAMLeSS


def pairwise_align(img_r, img_m, aligner, label=None):
    field = aligner.generate_field(img_r, img_m)
    img_w = aligner.warp_with_field(img_m, field.clone())
    if label is not None:
        label = aligner.warp_with_field(label, field.clone(), "nearest")
        return img_w, label
    else:
        return img_w


def batch_align(input_dir, aligner, output_dir):
    ncc_r = []
    ncc_g = []
    ncc_o = []  # img_m and img_g compute_ncc
    dice = []
    dice50 = []
    path_r_lst = glob.glob(os.path.join(input_dir, '*reference.tif'))
    path_m_lst = glob.glob(os.path.join(input_dir, "*moving.tif"))
    path_g_lst = glob.glob(os.path.join(input_dir, "*gt.tif"))
    path_l_lst = glob.glob(os.path.join(input_dir, "*label.tif"))
    path_lo_lst = glob.glob(os.path.join(input_dir, "*label_org.tif"))
    path_r_lst.sort()
    path_m_lst.sort()
    path_g_lst.sort()
    path_r_lst.sort()
    path_l_lst.sort()
    path_lo_lst.sort()
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for path_r, path_m, path_g, path_l, path_lo in zip(path_r_lst, path_m_lst, path_g_lst, path_l_lst, path_lo_lst):
        img_r = np.array(Image.open(path_r))
        img_m = np.array(Image.open(path_m))
        img_g = np.array(Image.open(path_g))
        img_l = np.array(Image.open(path_l))
        img_lo = np.array(Image.open(path_lo))
        img_w, img_l = pairwise_align(img_r, img_m, aligner, img_l)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(path_r).replace('reference', 'w')), img_w)
        shutil.copyfile(path_r, os.path.join(output_dir, os.path.basename(path_r).replace('reference', 'r')))
        shutil.copyfile(path_m, os.path.join(output_dir, os.path.basename(path_r).replace('reference', 'm')))
        shutil.copyfile(path_g, os.path.join(output_dir, os.path.basename(path_r).replace('reference', 'gt')))
        ncc_r.append(compute_ncc(img_r, img_w))
        ncc_g.append(compute_ncc(img_g, img_w))
        ncc_o.append(compute_ncc(img_m, img_g))
        dice.append(compute_dice_coefficients(img_lo, img_l))
        dice50.append(compute_dice_max_k(img_lo, img_l, 50))
    print("ncc between img_r and img_w: {:.3f}".format(np.mean(ncc_r)))
    print("ncc between img_g and img_w: {:.3f}".format(np.mean(ncc_g)))
    print("dice between img_g and img_w: {:.3f}".format(np.mean(dice)))
    print("dice50 between img_g and img_w: {:.3f}".format(np.mean(dice50)))
    return np.mean(ncc_r), np.mean(ncc_g)
    # print("ncc between img_g and img_m: {:.3f}".format(np.mean(ncc_o)))


def ext_call_pairwise(name, model_path, device):
    path = os.path.dirname(os.path.dirname(__file__))
    input_dir = "/media/ExtHDD01/xint/DATA/opticalData/CREMI_A/"
    output_dir = path + '/rst/{}'.format(name)
    is_c = True if name[-1] == 'c' else False
    aligner = FlowAligner(model_path, is_c, device)
    return batch_align(input_dir, aligner, output_dir)


# model_lst = {'ARFlow_ft': 'ARFlow_ft_4.pth', 'FE_ft': 'FE_ft_4.pth', 'ARFlow_ft_c': 'ARFlow_ft_c_9.pth',
#              'FE_ft_c': 'FE_ft_c_4.pth'}
model_lst = {'ARFlow_c': 'pwclite_ar.tar',
             'ARFlow_ft_c': 'ARFlow_ft_c_9.pth',
             'FE_ft_c': 'FE_pairwise.pth'}
if __name__ == "__main__":
    for name in model_lst:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_dir", type=str, default="/home/xint/mnt/DATA/opticalData/CREMI_A/")
        parser.add_argument("--model_path", type=str, default='../models/{}'.format(model_lst[name]))
        parser.add_argument("--output_dir", type=str, default='../rst/{}'.format(name))
        args = parser.parse_args()
        print("experiment {} start ...".format(name))
        is_c = True if name[-1] == 'c' else False
        aligner = FlowAligner(args.model_path, is_c, "cuda")
        batch_align(args.input_dir, aligner, args.output_dir)
    print("experiment {} start ...".format("SEAMLeSS"))
    aligner = SEAMLeSS()
    name = 'SEAMLeSS'
    args.output_dir = '../rst/{}'.format(name)
    batch_align(args.input_dir, aligner, args.output_dir)
