import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Experiment.estimate import compute_ncc


def compute_ncc_for_all_images(root1, root2, method):
    # 对 data_finished 文件夹下的每张图片计算 ncc 值
    ncc_values = []
    for filename in os.listdir(os.path.join(root1, 'pairwise', method, 'data_finished')):
        if filename.endswith('.png'):
            # 读取 data_finished 文件夹中的图片
            img1_path = os.path.join(root1, 'pairwise', method, 'data_finished', filename)
            img1 = np.array(Image.open(img1_path))

            # 找到 data_groundtruth 文件夹中与之对应的图片
            img2_path = os.path.join(root2, 'pairwise', 'data_groundtruth', filename)
            img2 = np.array(Image.open(img2_path))

            # 切成 64x64 的小块
            # 这里假设 img1 和 img2 的大小都是 64x64 的整数倍
            for i in range(0, img1.shape[0], 64):
                for j in range(0, img1.shape[1], 64):
                    # 计算 ncc 值
                    ncc = compute_ncc(img1[i:i+64, j:j+64], img2[i:i+64, j:j+64])
                    ncc_values.append(ncc)

    return ncc_values


def plot_ncc_values(ncc_values, method, index):
    # 绘制柱状图
    plt.bar(index, np.mean(ncc_values), width=0.2)
    plt.xlabel('Block index')
    plt.ylabel('NCC value')
    plt.title('NCC values for different methods')


def main(root1, root2, methods):
    # 初始化图像
    plt.figure()
    # 遍历所有的 method
    for i, method in enumerate(methods):
        ncc_values = compute_ncc_for_all_images(root1, root2, method)
        plot_ncc_values(ncc_values, method, i)
    plt.legend(methods)
    plt.savefig("test.png")
    # plt.show()
    

if __name__ == '__main__':
    # 设置 root1 和 root2 的路径
    root1 = '/home/xint/mnt/PyProjects/FESF/Experiment/Rst/CREMIA'
    root2 = '/home/xint/mnt/DATA/FESFData/CREMIA'
    methods = ['SIFTFlow', 'SEAMLeSS', 'FESF']
    main(root1, root2, methods)
    print("finished")