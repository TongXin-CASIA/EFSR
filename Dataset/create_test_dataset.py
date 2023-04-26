import glob
import os
import cv2
import shutil


def get_data(dir_org, dir_warped, dir_label, dir_label_o):
    path_o_lst = glob.glob(dir_org + "*.png")
    path_m_lst = glob.glob(dir_warped + "*.png")
    path_l_lst = glob.glob(dir_label + "*.png")
    path_lo_lst = glob.glob(dir_label_o + "*.png")
    path_o_lst.sort()
    path_m_lst.sort()
    path_l_lst.sort()
    path_lo_lst.sort()
    return path_o_lst, path_m_lst, path_l_lst, path_lo_lst


def create_pairwise_data(dir_org, dir_warped, dir_label, dir_label_o, dir_output):
    path_o_lst, path_m_lst, path_l_lst, path_lo_lst = get_data(dir_org, dir_warped, dir_label, dir_label_o)
    # mkdir output directory
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    for i in range(len(path_o_lst) - 1):
        # copy the file to the output directory
        img_r = cv2.imread(path_o_lst[i], cv2.IMREAD_GRAYSCALE)
        img_m = cv2.imread(path_m_lst[i + 1], cv2.IMREAD_GRAYSCALE)
        img_gt = cv2.imread(path_o_lst[i + 1], cv2.IMREAD_GRAYSCALE)
        img_l = cv2.imread(path_l_lst[i + 1], cv2.IMREAD_UNCHANGED)
        img_lo = cv2.imread(path_lo_lst[i + 1], cv2.IMREAD_UNCHANGED)
        # center crop 1024
        w, h = img_r.shape
        x = (w - 1024) // 2
        y = (h - 1024) // 2
        img_r = img_r[x:x + 1024, y:y + 1024]
        img_m = img_m[x:x + 1024, y:y + 1024]
        img_gt = img_gt[x:x + 1024, y:y + 1024]
        img_l = img_l[x:x + 1024, y:y + 1024]
        img_lo = img_lo[x:x + 1024, y:y + 1024]
        file_name = path_o_lst[i + 1].split('/')[-1].split('.')[0]
        cv2.imwrite(os.path.join(dir_output, file_name + "_reference.tif"), img_r)
        cv2.imwrite(os.path.join(dir_output, file_name + "_moving.tif"), img_m)
        cv2.imwrite(os.path.join(dir_output, file_name + "_gt.tif"), img_gt)
        cv2.imwrite(os.path.join(dir_output, file_name + "_label.tif"), img_l)
        cv2.imwrite(os.path.join(dir_output, file_name + "_label_org.tif"), img_lo)


def create_serial_data(dir_org, dir_warped, dir_label, dir_label_o, dir_output):
    path_o_lst, path_m_lst, path_l_lst, path_lo_lst = get_data(dir_org, dir_warped, dir_label, dir_label_o)
    # mkdir output directory
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    for i in range(32):
        if i == 0 or i == 31:
            img_o = cv2.imread(path_o_lst[i], cv2.IMREAD_GRAYSCALE)
            img_l = cv2.imread(path_lo_lst[i], cv2.IMREAD_UNCHANGED)
        else:
            img_o = cv2.imread(path_m_lst[i], cv2.IMREAD_GRAYSCALE)
            img_l = cv2.imread(path_l_lst[i], cv2.IMREAD_UNCHANGED)
        img_lo = cv2.imread(path_lo_lst[i], cv2.IMREAD_UNCHANGED)
        # center crop 1024
        w, h = img_o.shape
        x = (w - 1024) // 2
        y = (h - 1024) // 2
        img_l = img_l[x:x + 1024, y:y + 1024]
        img_o = img_o[x:x + 1024, y:y + 1024]
        img_lo = img_lo[x:x + 1024, y:y + 1024]
        file_name = path_o_lst[i].split('/')[-1].split('.')[0]
        cv2.imwrite(os.path.join(dir_output, file_name + "_serial.png"), img_o)
        cv2.imwrite(os.path.join(dir_output, file_name + "_label.png"), img_l)
        cv2.imwrite(os.path.join(dir_output, file_name + "_label_org.png"), img_lo)


def create_total_serial_data(dir_org, dir_warped, dir_label, dir_label_o, dir_output):
    path_o_lst, path_m_lst, path_l_lst, path_lo_lst = get_data(dir_org, dir_warped, dir_label, dir_label_o)
    # divided to 5 stacks
    for i in range(5):
        # path_out = os.path.join(dir_output, '{}'.format(i))
        path_out = dir_output
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        for j in range(25):
            idx = 25 * i + j
            if j == 0 or j == 24:
                img_o = cv2.imread(path_o_lst[idx], cv2.IMREAD_GRAYSCALE)
                img_l = cv2.imread(path_lo_lst[idx], cv2.IMREAD_UNCHANGED)
            else:
                img_o = cv2.imread(path_m_lst[idx], cv2.IMREAD_GRAYSCALE)
                img_l = cv2.imread(path_l_lst[idx], cv2.IMREAD_UNCHANGED)
            img_lo = cv2.imread(path_lo_lst[idx], cv2.IMREAD_UNCHANGED)
            # center crop 1024
            w, h = img_o.shape
            x = (w - 1024) // 2
            y = (h - 1024) // 2
            img_l = img_l[x:x + 1024, y:y + 1024]
            img_o = img_o[x:x + 1024, y:y + 1024]
            img_lo = img_lo[x:x + 1024, y:y + 1024]
            file_name = path_o_lst[idx].split('/')[-1].split('.')[0]
            cv2.imwrite(os.path.join(path_out, file_name + "_serial.png"), img_o)
            cv2.imwrite(os.path.join(path_out, file_name + "_label.png"), img_l)
            cv2.imwrite(os.path.join(path_out, file_name + "_label_org.png"), img_lo)


def serial_tiff2dir(path_input, dir_output, suffix):
    import skimage.io as io
    img = io.imread(path_input)
    im1 = img[0]
    l, t = im1.sum(0).nonzero()[0][0], im1.sum(1).nonzero()[0][0]
    img = img[:, t:t + 1024, l:l + 1024]
    os.makedirs(dir_output, exist_ok=True)
    for i, im in enumerate(img):
        cv2.imwrite(os.path.join(dir_output, '{:03d}'.format(i) + suffix), im)

def pairwise_for_fiji(dir_input):
    import numpy as np
    import skimage.io as io
    path_r_lst = glob.glob(dir_input + '*reference.tif')
    path_m_lst = glob.glob(dir_input + '*moving.tif')
    path_r_lst.sort()
    path_m_lst.sort()
    for path_r, path_m in zip(path_r_lst,path_m_lst):
        img = np.array([cv2.imread(path_r, cv2.IMREAD_GRAYSCALE), cv2.imread(path_m, cv2.IMREAD_GRAYSCALE)])
        io.imsave(path_m.replace('moving','pairwise'), img)


if __name__ == '__main__':
    create_pairwise_data(dir_org="/home/xint/mnt/DATA/CREMI_data/trainA/origin/",
                         dir_warped="/home/xint/mnt/DATA/CREMI_data/trainA/data_with_label/warped/",
                         dir_label="/home/xint/mnt/DATA/CREMI_data/trainA/data_with_label/warped_n_ids/",
                         dir_label_o="/home/xint/mnt/DATA/CREMI_data/trainA/data_with_label/label_org/",
                         dir_output="/home/xint/mnt/DATA/opticalData/CREMI_A/")
    # create_serial_data(dir_org="/home/xint/mnt/DATA/CREMI_data/trainA/origin/",
    #                    dir_warped="/home/xint/mnt/DATA/CREMI_data/trainA/data_with_label/warped/",
    #                    dir_label="/home/xint/mnt/DATA/CREMI_data/trainA/data_with_label/warped_n_ids/",
    #                    dir_label_o="/home/xint/mnt/DATA/CREMI_data/trainA/data_with_label/label_org/",
    #                    dir_output="/home/xint/mnt/DATA/opticalData/CREMI_A_Serial32/")
    # create_total_serial_data(dir_org="/home/xint/mnt/DATA/CREMI_data/trainA/origin/",
    #                          dir_warped="/home/xint/mnt/DATA/CREMI_data/trainA/data_with_label/warped/",
    #                          dir_label="/home/xint/mnt/DATA/CREMI_data/trainA/data_with_label/warped_n_ids/",
    #                          dir_label_o="/home/xint/mnt/DATA/CREMI_data/trainA/data_with_label/label_org/",
    #                          dir_output="/home/xint/mnt/DATA/opticalData/CREMI_A_Serial125/")
    # serial_tiff2dir("/media/ExtHDD01/xint/PyProjects/FESF/rst/CREMI_A_Serial32l.tif",
    #                 '/media/ExtHDD01/xint/PyProjects/FESF/rst/Elastic_s', suffix='_label.png')
    # pairwise_for_fiji("/media/ExtHDD01/xint/DATA/opticalData/CREMI_A/")
