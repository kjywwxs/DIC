"""
模拟散斑测试
"""
from DIC import DIC
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import time

ref_img = imread(
    'mydata/平移/Y1_6_0.bmp', '0')
if len(ref_img.shape) == 3:
    ref_img = ref_img.mean(axis=-1)
tar_img = imread(
    'mydata/平移/Y1_6_1.bmp', '0')
if len(tar_img.shape) == 3:
    tar_img = tar_img.mean(axis=-1)

# ref_img_dict = ['data/x y方向位移虚拟散斑/15.23pix-y方向位移/ImgSpeck1.bmp',
#                 'data/x y方向位移虚拟散斑/18.23pix-y方向 15.8pix-x方向/ImgSpeck1.bmp',]
# tar_img_dict = ['data/x y方向位移虚拟散斑/15.23pix-y方向位移/ImgSpeck2.bmp',
#                 'data/x y方向位移虚拟散斑/18.23pix-y方向 15.8pix-x方向/ImgSpeck2.bmp',]

# ref_img = imread(ref_img_dict[1], '0')
# tar_img = imread(tar_img_dict[1], '0')

dic = DIC(ref_img, tar_img)

params = {'subset_size': 31,
          'step': 5,
          'int_pixel_method': '逐点搜索',
          'sub_pixel_method': 'IC-GN',
          'ifauto': 1
          }
dic.set_parameters(**params)

fig, ax = plt.subplots(2, 2)
fig.set_size_inches(9, 7)
ax[0, 0].set_title('reference img')
ax[0, 0].imshow(ref_img, cmap='gray')
ax[0, 1].set_title('target img')
ax[0, 1].imshow(tar_img, cmap='gray')
# 在参考图像上选择计算点
dic.calculate_points(fig, ax[0, 0])

result = dic.start()

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

# # 结果图
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
