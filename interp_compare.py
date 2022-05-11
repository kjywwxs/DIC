import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from DIC import DIC

'''
比较
最近邻插值
双线性插值
双三次插值
双三次B样条插值
'''
# 生成原始灰度图像
np.random.seed(0)
gray_img = np.random.randint(0, 256, (5, 5)).astype('uint8')

# 计算插值后图像
dst1 = cv.resize(gray_img, (50, 50), interpolation=cv.INTER_NEAREST)

dst2 = cv.resize(gray_img, (50, 50), interpolation=cv.INTER_LINEAR)

dst3 = cv.resize(gray_img, (50, 50), interpolation=cv.INTER_CUBIC)

# 边上那一圈不知道咋算,先向外延展一像素
gray_img_big = cv.copyMakeBorder(gray_img, 1, 1, 1, 1, cv.BORDER_REPLICATE)
a = np.arange(1, 6, 0.1)
b = np.arange(1, 6, 0.1)
PMeshX, PMeshY = np.meshgrid(a, b, indexing='ij')
cal_points = np.concatenate((PMeshX.reshape(-1, 1), PMeshY.reshape(-1, 1)), axis=1).T
cal_points = np.vstack((cal_points, np.ones(2500)))
dst4 = DIC.bicubic_Bspline_interp(gray_img_big, cal_points)
dst4 = dst4.reshape(50, 50)

# 原
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.set_xlabel('X')
ax.set_ylabel('Y')
x = np.arange(0, 5)
y = np.arange(0, 5)
X, Y = np.meshgrid(x, y)
Z = gray_img
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

# 最近邻插值
fig2, ax2 = plt.subplots(subplot_kw=dict(projection='3d'))
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
x = np.arange(0, 5, 0.1)
y = np.arange(0, 5, 0.1)
X, Y = np.meshgrid(x, y)
Z = dst1
ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

# 双线性插值
fig3, ax3 = plt.subplots(subplot_kw=dict(projection='3d'))
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
x = np.arange(0, 5, 0.1)
y = np.arange(0, 5, 0.1)
X, Y = np.meshgrid(x, y)
Z = dst2
ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

# 双三次插值
fig4, ax4 = plt.subplots(subplot_kw=dict(projection='3d'))
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
x = np.arange(0, 5, 0.1)
y = np.arange(0, 5, 0.1)
X, Y = np.meshgrid(x, y)
Z = dst3
ax4.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

# 双三次样条插值
fig5, ax5 = plt.subplots(subplot_kw=dict(projection='3d'))
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
x = np.arange(0, 5, 0.1)
y = np.arange(0, 5, 0.1)
X, Y = np.meshgrid(x, y)
Z = dst4
ax5.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()
