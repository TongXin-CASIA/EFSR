import os
import shutil

import cv2
import glob


def pairwise_data(dirname, stripe=1):
    """
    params: dir_input
    params: dir_output
    """
    # dir_in
    if True:
        dir_origin_data = os.path.join(dirname, "origin", "data")
        dir_origin_label = os.path.join(dirname, "origin", "label")
        dir_warpped_data = os.path.join(dirname, "warped", "data")
        dir_warpped_label = os.path.join(dirname, "warped", "label")
        path_origin_data = glob.glob(dir_origin_data + "/*.png")
        path_origin_label = glob.glob(dir_origin_label + "/*.png")
        path_warpped_data = glob.glob(dir_warpped_data + "/*.png")
        path_warpped_label = glob.glob(dir_warpped_label + "/*.png")
        path_origin_data.sort()
        path_origin_label.sort()
        path_warpped_data.sort()
        path_warpped_label.sort()
    # 生成文件夹 dir_out
    if True:
        # dir_out
        dir_data_reference = os.path.join(dirname, "pairwise", "data_reference")
        dir_data_moving = os.path.join(dirname, "pairwise", "data_moving")
        dir_data_groundtruth = os.path.join(dirname, "pairwise", "data_groundtruth")
        dir_label_moving = os.path.join(dirname, "pairwise", "label_moving")
        dir_label_groundtruth = os.path.join(dirname, "pairwise", "label_groundtruth")
        os.makedirs(dir_data_reference, exist_ok=True)
        os.makedirs(dir_data_moving, exist_ok=True)
        os.makedirs(dir_data_groundtruth, exist_ok=True)
        os.makedirs(dir_label_moving, exist_ok=True)
        os.makedirs(dir_label_groundtruth, exist_ok=True)
    for i in range(0, len(path_origin_data) - stripe, stripe):
        data_reference = path_origin_data[i]
        data_moving = path_warpped_data[i + stripe]
        data_groundtruth = path_origin_data[i + stripe]
        if path_warpped_label:
            label_moving = path_warpped_label[i + stripe]
            label_groundtruth = path_origin_label[i + stripe]
        img_r = cv2.imread(data_reference, cv2.IMREAD_GRAYSCALE)
        w, h = img_r.shape
        x = (w - 1024) // 2
        y = (h - 1024) // 2
        img_reference = cv2.imread(data_reference, cv2.IMREAD_GRAYSCALE)[x:x + 1024, y:y + 1024]
        img_moving = cv2.imread(data_moving, cv2.IMREAD_GRAYSCALE)[x:x + 1024, y:y + 1024]
        img_groundtruth = cv2.imread(data_groundtruth, cv2.IMREAD_GRAYSCALE)[x:x + 1024, y:y + 1024]
        if path_warpped_label:
            img_label_moving = cv2.imread(label_moving, cv2.IMREAD_UNCHANGED)[x:x + 1024, y:y + 1024]
            img_label_groundtruth = cv2.imread(label_groundtruth, cv2.IMREAD_UNCHANGED)[x:x + 1024, y:y + 1024]
        cv2.imwrite(data_reference.replace(dir_origin_data, dir_data_reference), img_reference)
        cv2.imwrite(data_reference.replace(dir_origin_data, dir_data_moving), img_moving)
        cv2.imwrite(data_reference.replace(dir_origin_data, dir_data_groundtruth), img_groundtruth)
        if path_warpped_label:
            cv2.imwrite(data_reference.replace(dir_origin_data, dir_label_moving), img_label_moving)
            cv2.imwrite(data_reference.replace(dir_origin_data, dir_label_groundtruth), img_label_groundtruth)

        # shutil.copyfile(data_reference, data_reference.replace(dir_origin_data, dir_data_reference))
        # shutil.copyfile(data_moving, data_reference.replace(dir_origin_data, dir_data_moving))
        # shutil.copyfile(data_groundtruth, data_reference.replace(dir_origin_data, dir_data_groundtruth))
        # shutil.copyfile(label_moving, data_reference.replace(dir_origin_data, dir_label_moving))
        # shutil.copyfile(label_groundtruth, data_reference.replace(dir_origin_data, dir_label_groundtruth))


def serial32_data(dirname, stripe=1):
    # <editor-fold desc="prepare dir_in">
    dir_origin_data = os.path.join(dirname, "origin", "data")
    dir_origin_label = os.path.join(dirname, "origin", "label")
    dir_warpped_data = os.path.join(dirname, "warped", "data")
    dir_warpped_label = os.path.join(dirname, "warped", "label")
    path_origin_data = glob.glob(dir_origin_data + "/*.png")
    path_origin_label = glob.glob(dir_origin_label + "/*.png")
    path_warpped_data = glob.glob(dir_warpped_data + "/*.png")
    path_warpped_label = glob.glob(dir_warpped_label + "/*.png")
    path_origin_data.sort()
    path_origin_label.sort()
    path_warpped_data.sort()
    path_warpped_label.sort()
    # </editor-fold>

    # <editor-fold desc="prepare dir_out">
    try:
        shutil.rmtree(os.path.join(dirname, "serial32"))
    except:
        pass
    dir_data = os.path.join(dirname, "serial32", "data")
    dir_label = os.path.join(dirname, "serial32", "label")
    dir_data_groundtruth = os.path.join(dirname, "serial32", "data_groundtruth")
    dir_label_groundtruth = os.path.join(dirname, "serial32", "label_groundtruth")
    os.makedirs(dir_data, exist_ok=True)
    os.makedirs(dir_label, exist_ok=True)
    os.makedirs(dir_data_groundtruth, exist_ok=True)
    os.makedirs(dir_label_groundtruth, exist_ok=True)
    # </editor-fold>

    for i in range(0, len(path_origin_data) - stripe, stripe):
        if i > 31*stripe:
            break
        if i == 0 or i == 31:
            dir_input_data = dir_origin_data
            path_data = path_origin_data
            path_label = path_origin_label
        else:
            dir_input_data = dir_warpped_data
            path_data = path_warpped_data
            path_label = path_warpped_label
        data = path_data[i]
        data_groundtruth = path_origin_data[i]
        if path_warpped_label:
            label = path_label[i]
            label_groundtruth = path_origin_label[i]
        img_r = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
        w, h = img_r.shape
        x = (w - 1024) // 2
        y = (h - 1024) // 2
        img = cv2.imread(data, cv2.IMREAD_GRAYSCALE)[x:x + 1024, y:y + 1024]
        img_groundtruth = cv2.imread(data_groundtruth, cv2.IMREAD_GRAYSCALE)[x:x + 1024, y:y + 1024]
        cv2.imwrite(data.replace(dir_input_data, dir_data), img)
        cv2.imwrite(data.replace(dir_input_data, dir_data_groundtruth), img_groundtruth)
        if path_warpped_label:
            img_label = cv2.imread(label, cv2.IMREAD_UNCHANGED)[x:x + 1024, y:y + 1024]
            img_label_groundtruth = cv2.imread(label_groundtruth, cv2.IMREAD_UNCHANGED)[x:x + 1024, y:y + 1024]
            cv2.imwrite(data.replace(dir_input_data, dir_label), img_label)
            cv2.imwrite(data.replace(dir_input_data, dir_label_groundtruth), img_label_groundtruth)


if __name__ == "__main__":
    # pairwise_data("/home/xint/mnt/DATA/FESFData/CREMIA/")
    pairwise_data("/home/xint/mnt/DATA/FESFData/FIB_mito/", 8)
    # serial32_data("/home/xint/mnt/DATA/FESFData/CREMIA/")
    serial32_data("/home/xint/mnt/DATA/FESFData/FIB_mito/", 8)
    print("finished")
