import numpy as np


def compute_ncc(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img1 = (img1 - img1.mean()) / img1.std()
    img2 = (img2 - img2.mean()) / img2.std()
    return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def compute_dice_coefficients(label1, label2):
    labels = np.unique(label1)
    dice = 0
    for l in labels:
        dice += dice_coef(label1 == l, label2 == l)
    return dice / len(labels)


def compute_dice_max_k(label1, label2, k):
    labels = np.unique(label1)
    dice = []
    # get max k values
    area = []
    for l in labels:
        area.append(((label1 == l).sum(), l))
    area.sort(key=lambda x: x[0], reverse=True)
    for a, l in area[:k]:
        dice.append(dice_coef(label1 == l, label2 == l))
    return np.mean(dice)


if __name__ == '__main__':
    import cv2
    import glob

    dir_label = "/home/xint/mnt/DATA/CREMI_data/trainA/data_with_label/warped_n_ids/"
    dir_label_o = "/home/xint/mnt/DATA/CREMI_data/trainA/data_with_label/label_org/"
    path_l_lst = glob.glob(dir_label + "*.png")
    path_lo_lst = glob.glob(dir_label_o + "*.png")
    path_l_lst.sort()
    path_lo_lst.sort()
    for plo, pl in zip(path_lo_lst, path_l_lst):
        label1 = cv2.imread(plo, cv2.IMREAD_UNCHANGED)
        label2 = cv2.imread(pl, cv2.IMREAD_UNCHANGED)
        print(compute_dice_coefficients(label1, label2))
