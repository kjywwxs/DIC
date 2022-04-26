import cv2
import numpy as np
import matplotlib.image as mpimg

'''
这种方法不能准确利用两幅图像所有信息，尤其是有亚像素位移时准确度低
'''


# 均值哈希算法
def aHash(img):
    # 缩放为8*8
    img = cv2.resize(img, (8, 8))
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + img[i, j]
    # 求平均灰度
    avg = s / 64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if img[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 差值感知算法
def dHash(img):
    img = cv2.resize(img, (9, 8))
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if img[i, j] > img[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 感知哈希算法(pHash)
def pHash(img):
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(img))
    # opencv实现的掩码操作
    dct_roi = dct[0:8, 0:8]

    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


# Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


if __name__ == '__main__':
    # ref_img = mpimg.imread(
    #     'D:/桌面/毕设/pictures/2_0.bmp', '0')
    # ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    # tar_img = mpimg.imread(
    #     'D:/桌面/毕设/pictures/2_1.bmp', '0')
    # tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2GRAY)

    ref_img_dict = ['data/x y方向位移虚拟散斑/15.23pix-y方向位移/ImgSpeck1.bmp',
                    'data/x y方向位移虚拟散斑/18.23pix-y方向 15.8pix-x方向/ImgSpeck1.bmp']
    tar_img_dict = ['data/x y方向位移虚拟散斑/15.23pix-y方向位移/ImgSpeck2.bmp',
                    'data/x y方向位移虚拟散斑/18.23pix-y方向 15.8pix-x方向/ImgSpeck2.bmp']

    ref_img = mpimg.imread(ref_img_dict[0], '0')
    tar_img = mpimg.imread(tar_img_dict[0], '0')

    focus_point = np.array([100, 120, 1])
    subset_size = 31
    fSubset = ref_img[100 - 15:100 + 15 + 1, 120 - 15:120 + 15 + 1]

    # sizeX = np.size(ref_img, 0)
    # sizeY = np.size(ref_img, 1)
    # step = 1
    # x = np.arange(0, sizeX - subset_size + 1, step) + subset_size // 2
    # y = np.arange(0, sizeY - subset_size + 1, step) + subset_size // 2
    # X, Y = np.meshgrid(x, y)
    # # 所有位移后的点的坐标
    # all_xy = np.vstack((X.flatten('F'), Y.flatten('F'))).T.astype('int32')

    all_xy = np.array([[99, 170],
                       [100, 135],
                       [100, 136]])

    i = 1
    gSubset = tar_img[all_xy[i, 0] - 15:all_xy[i, 0] + 15 + 1, all_xy[i, 1] - 15:all_xy[i, 1] + 15 + 1]
    hash1 = aHash(fSubset)
    hash2 = aHash(gSubset)
    n = cmpHash(hash1, hash2)
    print('均值哈希算法相似度：', n)

    hash1 = dHash(fSubset)
    hash2 = dHash(gSubset)
    n = cmpHash(hash1, hash2)
    print('差值哈希算法相似度：', n)

    hash1 = pHash(fSubset)
    hash2 = pHash(gSubset)
    n = cmpHash(hash1, hash2)
    print('感知哈希算法相似度：', n)
