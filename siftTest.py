import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.image as mpimg
import math

if __name__ == '__main__':
    img1 = mpimg.imread(
        'mydata/旋转/默认45度_0.bmp', '0')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = mpimg.imread(
        'mydata/旋转/默认45度_1.bmp', '0')
    img2 = cv2.resize(img2, (360, 360), interpolation=cv2.INTER_CUBIC)
    img2 = img2[30:330, 30:330, :]
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)  # des是描述子
    kp2, des2 = sift.detectAndCompute(img2, None)  # des是描述子

    hmerge = np.hstack((img1, img2))  # 水平拼接
    cv2.imshow("point", hmerge)  # 拼接显示为gray
    cv2.waitKey(0)

    img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
    img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈

    hmerge = np.hstack((img3, img4))  # 水平拼接
    cv2.imshow("point", hmerge)  # 拼接显示为gray
    cv2.waitKey(0)

    # BFMatcher解决匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append([m])

    img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    cv2.imshow("BFmatch", img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    j = 0
    for i in good:
        # 获取匹配的特征点坐标
        xy1 = kp1[i[0].queryIdx].pt
        xy2 = kp2[i[0].trainIdx].pt
        y1_pie = 1.2 * ((xy1[1] - 150) * math.cos(-math.pi / 4) - (xy1[0] - 150) * math.sin(-math.pi / 4)) + 150
        x1_pie = 1.2 * ((xy1[1] - 150) * math.sin(-math.pi / 4) + (xy1[0] - 150) * math.cos(-math.pi / 4)) + 150
        judge = math.sqrt(
            (x1_pie - xy2[0]) * (x1_pie - xy2[0]) + (y1_pie - xy2[1]) * (y1_pie - xy2[1]))
        if judge < 1:
            j = j + 1
    # 计算匹配准确的点的百分比
    print(j/len(good))
