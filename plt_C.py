import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg

if __name__ == '__main__':
    # ref_img_dict = ['data/x y方向位移虚拟散斑/15.23pix-y方向位移/ImgSpeck1.bmp',
    #                 'data/x y方向位移虚拟散斑/18.23pix-y方向 15.8pix-x方向/ImgSpeck1.bmp']
    # tar_img_dict = ['data/x y方向位移虚拟散斑/15.23pix-y方向位移/ImgSpeck2.bmp',
    #                 'data/x y方向位移虚拟散斑/18.23pix-y方向 15.8pix-x方向/ImgSpeck2.bmp']
    #
    # ref_img = mpimg.imread(ref_img_dict[0], '0')
    # tar_img = mpimg.imread(tar_img_dict[0], '0')
    ref_img = mpimg.imread(
        'D:/桌面/毕设/pictures/3_0.bmp', '0')
    ref_img = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
    tar_img = mpimg.imread(
        'D:/桌面/毕设/pictures/3_1.bmp', '0')
    tar_img = cv.cvtColor(tar_img, cv.COLOR_BGR2GRAY)

    sizeX = np.size(ref_img, 0)
    sizeY = np.size(ref_img, 1)
    subset_size = 31
    half_subset = int((subset_size - 1) / 2)
    # 参考图像相关数据
    focus_point = np.array([100, 120, 1])

    fSubset = ref_img[100 - half_subset:100 + half_subset + 1, 120 - half_subset:120 + half_subset + 1]

    deltafVec = fSubset - np.mean(fSubset)
    deltaf = np.sqrt(np.sum(deltafVec ** 2))
    # 不减均值，用于计算NCC,NSSD
    deltaf_N = np.sqrt(np.sum(fSubset ** 2))

    step = 1
    # x = np.arange(0, sizeX - subset_size + 1, step) + subset_size // 2
    # y = np.arange(0, sizeY - subset_size + 1, step) + subset_size // 2
    x = np.arange(50, 250 - subset_size + 1, step) + subset_size // 2
    y = np.arange(50, 250 - subset_size + 1, step) + subset_size // 2
    X, Y = np.meshgrid(x, y)
    # 所有位移后的点的坐标
    all_xy = np.vstack((X.flatten('F'), Y.flatten('F'))).T.astype('int32')

    # Ccc = np.zeros(len(all_xy))
    # Cncc = np.zeros(len(all_xy))
    # Czncc = np.zeros(len(all_xy))
    # Cssd = np.zeros(len(all_xy))
    # Cnssd = np.zeros(len(all_xy))
    Cznssd = np.zeros(len(all_xy))
    for i in range(len(all_xy)):
        current_gSubset = tar_img[all_xy[i, 0] - half_subset:all_xy[i, 0] + half_subset + 1,
                          all_xy[i, 1] - half_subset:all_xy[i, 1] + half_subset + 1]

        # Ccc[i] = np.sum(fSubset*current_gSubset)
        # '''ncc有疑问'''
        # deltag_N = np.sqrt(np.sum(current_gSubset ** 2))
        # Cncc[i] = np.sum(fSubset * current_gSubset)/deltag_N/deltaf_N

        # deltagVec = current_gSubset - np.mean(current_gSubset)
        # deltag = np.sqrt(np.sum(deltagVec ** 2))
        # Czncc[i] = np.sum(deltafVec*deltagVec)/deltaf/deltag

        # Cssd[i] = np.sum((fSubset-current_gSubset)**2)

        # deltag_N = np.sqrt(np.sum(current_gSubset ** 2))
        # Cnssd[i] = np.sum((fSubset-current_gSubset)**2)/deltag_N/deltaf_N

        deltagVec = current_gSubset - np.mean(current_gSubset)
        deltag = np.sqrt(np.sum(deltagVec ** 2))
        Cznssd[i] = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    shape = np.shape(X)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('C')
    # X,Y位置调换只是索引的问题
    # Z = Ccc.reshape(shape)
    # Z = Cncc.reshape(shape)
    # Z = Czncc.reshape(shape)
    Z = Cznssd.reshape(shape)
    # 反向z轴
    ax.invert_zaxis()
    ax.plot_surface(Y, X, Z, rstride=1, cstride=1, cmap='rainbow')

    # fig, ax = plt.subplots(3, 2, subplot_kw=dict(projection='3d'))
    # shape = np.shape(X)
    # ax[0, 0].set_xlabel('X')
    # ax[0, 0].set_ylabel('Y')
    # # X,Y位置调换只是索引的问题
    # Z = Ccc.reshape(shape)
    # ax[0, 0].plot_surface(Y, X, Z, rstride=1, cstride=1, cmap='rainbow')
    # ax[0, 1].set_xlabel('X')
    # ax[0, 1].set_ylabel('Y')
    # Z = Cncc.reshape(shape)
    # ax[0, 1].plot_surface(Y, X, Z, rstride=1, cstride=1, cmap='rainbow')
    # ax[1, 0].set_xlabel('X')
    # ax[1, 0].set_ylabel('Y')
    # Z = Czncc.reshape(shape)
    # ax[1, 0].plot_surface(Y, X, Z, rstride=1, cstride=1, cmap='rainbow')
    # ax[1, 1].set_xlabel('X')
    # ax[1, 1].set_ylabel('Y')
    # Z = Cssd.reshape(shape)
    # ax[1, 1].plot_surface(Y, X, Z, rstride=1, cstride=1, cmap='rainbow')
    # ax[2, 0].set_xlabel('X')
    # ax[2, 0].set_ylabel('Y')
    # Z = Cnssd.reshape(shape)
    # ax[2, 0].plot_surface(Y, X, Z, rstride=1, cstride=1, cmap='rainbow')
    # ax[2, 1].set_xlabel('X')
    # ax[2, 1].set_ylabel('Y')
    # Z = Cznssd.reshape(shape)
    # ax[2, 1].plot_surface(Y, X, Z, rstride=1, cstride=1, cmap='rainbow')

    plt.show()
