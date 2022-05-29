"""
模拟散斑测试
"""
from DIC import DIC
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2 as cv
import numpy as np
import math
import time
from rand_elastic_distortion import elastic_transform


def add_light(img):
    '''给图像施加不均匀光照，输入为np数组的二维图片'''
    X = np.size(img, 0)
    Y = np.size(img, 1)
    strength = 20  # 设置光照强度
    center_point = np.array([150, 100])  # 设置光照中心点
    radius = np.min(center_point)
    for i in range(X):
        for j in range(Y):
            distance = math.pow((center_point[0] - j), 2) + math.pow((center_point[1] - i), 2)
            if distance < radius * radius:
                result = int(strength * (1.0 - math.sqrt(distance) / radius))
                temp = img[i, j] + result
                # 防止越界
                img[i, j] = min(255, max(0, temp))
    return img


def add_noise(img, mean=0, var=0.001):
    '''
    添加高斯噪声
    mean : 均值
    var : 方差
    '''
    img = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out = img + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)

    return out


def calpt_pingyi(x):
    d = 1.6
    return x + d


def calpt_yasuo(x):
    '''
    输入压缩前点的y坐标
    返回压缩后点的y坐标
    '''
    a = -0.3
    # glare软件生成位移图时像素坐标索引和python中差了1
    x0 = 150 + 1
    return a * (x - x0) + x


def calpt_zhenxian(x):
    '''
    输入正弦变形前点的y坐标
    返回正弦变形后点的y坐标
    '''
    a = 1
    T = 100
    b = 0
    return a * math.sin(2 * math.pi * x / T + b) + x


def calpt_gaosi(x):
    '''
    输入高斯变形前点的y坐标
    返回高斯变形后点的y坐标
    '''
    a = 4
    x0 = 150
    c = 15
    return a * math.exp(-(x - x0) * (x - x0) / c / c) + x


def calpt_xuanzhuan(x):
    '''
    输入旋转前点的坐标（numpy数组）,N×2
    返回旋转后点的坐标（numpy数组）,N×2
    '''
    a = 1
    center = np.array([151, 151])
    thet = -math.pi / 180 * 32
    # 变换矩阵
    M = np.array([[math.cos(thet), -math.sin(thet)],
                  [math.sin(thet), math.cos(thet)]])
    temp = (x - center).T
    return (a * M @ temp).T + center


def cal_err_sd(xy_disp, ref_pt, def_type):
    '''
    xy_disp为测量出的点xy方向位移
    ref_pt为点初始位置
    def_type指变形类型0：平移，1：压缩，2：正弦，3：高斯，4：旋转
    '''
    # 计算测量得到的位移后位置
    x_c = ref_pt[:, 0] + xy_disp[:, 0]
    y_c = ref_pt[:, 1] + xy_disp[:, 1]
    # 计算出理论真实值
    if def_type == 0:
        # 把方法转成矢量形式
        calpt_pingyi_v = np.vectorize(calpt_pingyi)
        x_moved = ref_pt[:, 0]
        y_moved = calpt_pingyi_v(ref_pt[:, 1])
    elif def_type == 1:
        calpt_yasuo_v = np.vectorize(calpt_yasuo)
        x_moved = ref_pt[:, 0]
        y_moved = calpt_yasuo_v(ref_pt[:, 1])
    elif def_type == 2:
        calpt_zhenxian_v = np.vectorize(calpt_zhenxian)
        x_moved = ref_pt[:, 0]
        y_moved = calpt_zhenxian_v(ref_pt[:, 1])
    elif def_type == 3:
        calpt_gaosi_v = np.vectorize(calpt_gaosi)
        x_moved = ref_pt[:, 0]
        y_moved = calpt_gaosi_v(ref_pt[:, 1])
    else:
        xy_moved = calpt_xuanzhuan(ref_pt)
        x_moved = xy_moved[:, 0]
        y_moved = xy_moved[:, 1]

    # 如果这n个点不属于对同一位移的同一测量，则标准差没意义
    x_sd = math.sqrt(sum((xy_disp[:, 0] - xy_disp[:, 0].mean()) ** 2) / (len(xy_disp[:, 0]) - 1))
    y_sd = math.sqrt(sum((xy_disp[:, 1] - xy_disp[:, 1].mean()) ** 2) / (len(xy_disp[:, 1]) - 1))

    return x_c - x_moved, x_sd, y_c - y_moved, y_sd


if __name__ == '__main__':
    ref_img_dict = ['mydata/平移/Y1_6_0.bmp',
                    'mydata/拉伸/a负0_1_0.bmp',
                    'mydata/正弦/a1T50_0.bmp',
                    'mydata/高斯/a1c15_0.bmp',
                    'mydata/旋转/10度_0.bmp']
    tar_img_dict = ['mydata/平移/Y1_6_1.bmp',
                    'mydata/拉伸/a负0_1_1.bmp',
                    'mydata/正弦/a1T50_1.bmp',
                    'mydata/高斯/a1c15_1.bmp',
                    'mydata/旋转/10度_1.bmp']
    # 0：平移，1：压缩，2：正弦，3：高斯，4：旋转
    def_type = 0
    ref_img = imread('mydata/平移/Y1_6_0.bmp', '0')
    if len(ref_img.shape) == 3:
        ref_img = ref_img.mean(axis=-1)
    # tar_img = imread('mydata/平移/Y1_6_1.bmp', '0')
    tar_img, M, dx, dy = elastic_transform(ref_img,
                                           ref_img.shape[1] * 0.32,
                                           ref_img.shape[1] * 0.08,
                                           ref_img.shape[1] * 0.01,
                                           np.random.RandomState())
    if len(tar_img.shape) == 3:
        tar_img = tar_img.mean(axis=-1)

    # ref_img = add_noise(ref_img, mean=0, var=0.001)  # 施加噪声
    # tar_img = add_noise(tar_img, mean=0, var=0.001)  # 施加噪声
    # ref_img = cv.GaussianBlur(ref_img, (3, 3), 1)  # 高斯滤波
    # tar_img = cv.GaussianBlur(tar_img, (3, 3), 1)  # 高斯滤波

    dic = DIC(ref_img, tar_img)
    # int_pixel_method：给定为零；逐点搜索；十字搜索；粗细十字搜索；GA;手动给定
    params = {'subset_size': 31,
              'step': 1,
              'int_pixel_method': '逐点搜索',
              'sub_pixel_method': 'IC-GN2',
              'ifauto': 1
              }
    dic.set_parameters(**params)
    # 绘制参考图像和目标图像
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(9, 7)
    ax[0, 0].set_title('reference img')
    ax[0, 0].imshow(ref_img, cmap='gray')
    ax[0, 1].set_title('target img')
    ax[0, 1].imshow(tar_img, cmap='gray')
    # 在参考图像上选择计算点
    dic.calculate_points(fig, ax[0, 0])
    result = dic.start()

    # # 计算误差
    cal_points = dic.cal_points
    # x_err, x_sd, y_err, y_sd = cal_err_sd(result[0], cal_points, def_type)

    # 计算随机弹性形变误差 坐标系不太一样，有点晕
    cal_points[:, [0, 1]] = cal_points[:, [1, 0]]
    fangshehou = (M[:, 0:2] @ cal_points.T + M[:, 2].reshape(2, 1)).T
    fangshehou[:, [0, 1]] = fangshehou[:, [1, 0]]

    cal_dy = dx[dic.cal_pointsX][:, dic.cal_pointsY].flatten()
    cal_dx = dy[dic.cal_pointsX][:, dic.cal_pointsY].flatten()
    xy_moved = fangshehou + np.vstack((cal_dx, cal_dy)).T

    cal_points[:, [0, 1]] = cal_points[:, [1, 0]]
    x_c = cal_points[:, 0] + result[0][:, 0]
    y_c = cal_points[:, 1] + result[0][:, 1]

    x_err = x_c - xy_moved[:, 0]
    y_err = y_c - xy_moved[:, 1]
    x_sd = math.sqrt(sum((result[0][:, 0] - result[0][:, 0].mean()) ** 2) / (len(result[0][:, 0]) - 1))
    y_sd = math.sqrt(sum((result[0][:, 1] - result[0][:, 1].mean()) ** 2) / (len(result[0][:, 1]) - 1))

    # 绘制误差分布图
    x_err_plt = x_err.reshape(dic.Lx, dic.Ly)
    y_err_plt = y_err.reshape(dic.Lx, dic.Ly)
    temp = 5
    fig3, ax3 = plt.subplots(1, 2)
    ax3[0].set_title('X err')
    A = ax3[0].imshow(x_err_plt, cmap='nipy_spectral')
    fig3.colorbar(A, ax=ax3[0])
    ax3[0].set_xticks(np.arange(0, dic.Ly, temp), np.arange(0, dic.Ly * dic.step, temp * dic.step))
    ax3[0].set_yticks(np.arange(0, dic.Lx, temp), np.arange(0, dic.Lx * dic.step, temp * dic.step))
    ax3[1].set_title('Y err')
    B = ax3[1].imshow(y_err_plt, cmap='nipy_spectral')
    fig.colorbar(B, ax=ax3[1])
    ax3[1].set_xticks(np.arange(0, dic.Ly, temp), np.arange(0, dic.Ly * dic.step, temp * dic.step))
    ax3[1].set_yticks(np.arange(0, dic.Lx, temp), np.arange(0, dic.Lx * dic.step, temp * dic.step))

    # 绘制结果图像
    xy = result[0]
    x = xy[:, 0]
    x = x.reshape(dic.Lx, dic.Ly)
    y = xy[:, 1]
    y = y.reshape(dic.Lx, dic.Ly)
    temp = 5  # 控制刻度

    ax[1, 0].set_title('X displacement')
    A = ax[1, 0].imshow(x, cmap='nipy_spectral')
    fig.colorbar(A, ax=ax[1, 0])
    ax[1, 0].set_xticks(np.arange(0, dic.Ly, temp), np.arange(0, dic.Ly * dic.step, temp * dic.step))
    ax[1, 0].set_yticks(np.arange(0, dic.Lx, temp), np.arange(0, dic.Lx * dic.step, temp * dic.step))

    ax[1, 1].set_title('Y displacement')
    B = ax[1, 1].imshow(y, cmap='nipy_spectral')
    fig.colorbar(B, ax=ax[1, 1])
    ax[1, 1].set_xticks(np.arange(0, dic.Ly, temp), np.arange(0, dic.Ly * dic.step, temp * dic.step))
    ax[1, 1].set_yticks(np.arange(0, dic.Lx, temp), np.arange(0, dic.Lx * dic.step, temp * dic.step))

    # 单结果图
    # fig2, ax2 = plt.subplots(1, 2)
    # ax2[0].set_title('X displacement')
    # A = ax2[0].imshow(x, cmap='nipy_spectral')
    # fig2.colorbar(A, ax=ax2[0])
    # ax2[0].set_xticks(np.arange(0, dic.Ly, temp), np.arange(0, dic.Ly * dic.step, temp * dic.step))
    # ax2[0].set_yticks(np.arange(0, dic.Lx, temp), np.arange(0, dic.Lx * dic.step, temp * dic.step))
    # ax2[1].set_title('Y displacement')
    # B = ax2[1].imshow(y, cmap='nipy_spectral')
    # fig2.colorbar(B, ax=ax2[1])
    # ax2[1].set_xticks(np.arange(0, dic.Ly, temp), np.arange(0, dic.Ly * dic.step, temp * dic.step))
    # ax2[1].set_yticks(np.arange(0, dic.Lx, temp), np.arange(0, dic.Lx * dic.step, temp * dic.step))

    plt.show()
