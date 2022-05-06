import cv2 as cv
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import scipy.signal
import find_point_GA


class DIC:

    def __init__(self, ref_img, tar_img, debug=False):
        self.ref_img = ref_img
        self.tar_img = tar_img
        # 是否自动选取区域
        self.ifauto = 1
        self.sizeX = np.size(ref_img, 0)
        self.sizeY = np.size(ref_img, 1)
        self.debug = debug

    def set_parameters(self, subset_size: int = 31):
        self.subset_size = subset_size
        self.step = 5
        # 迭代停止条件
        self.thre = 1e-3
        # 为1则进行归一化，为0则不进行归一化
        self.Normalization = 1
        # 最大迭代次数
        self.max_iter = 15
        # 图像子区局部坐标
        self.half_subset = math.floor(subset_size / 2)
        self.deltaVecX = np.arange(-self.half_subset, self.half_subset + 1)
        self.deltaVecY = np.arange(-self.half_subset, self.half_subset + 1)
        deltax, deltay = np.meshgrid(self.deltaVecX, self.deltaVecY, indexing='xy')
        self.localSubHom = np.concatenate(
            (deltax.reshape(1, -1), deltay.reshape(1, -1), np.ones((1, subset_size * subset_size))), axis=0)
        self.localSub = self.localSubHom[0:2].T
        self.sub_pixel_method = 'IC-GN'

    def calculate_points(self):
        half_subset = self.half_subset
        step = self.step
        x = np.zeros(2)
        y = np.zeros(2)
        if self.ifauto:
            # 预选取的感兴趣范围
            x = [100, self.sizeX - 100]
            y = [100, self.sizeY - 100]
        else:
            # 手动给定
            temp = plt.ginput(1)
            y[0] = temp[0][0]
            x[0] = temp[0][1]
            plt.plot(y[0], x[0], '+r', 8)
            plt.show()
            temp = plt.ginput(1)
            y[1] = temp[0][0]
            x[1] = temp[0][1]
            plt.plot(y[1], x[1], '+r', 8)
            plt.show()

        x = np.round(x)
        y = np.round(y)
        xROI = np.sort(x)
        yROI = np.sort(y)

        plt.plot([yROI[0], yROI[0]], [xROI[0], xROI[1]], '-r', 1)
        plt.plot([yROI[1], yROI[1]], [xROI[0], xROI[1]], '-r', 1)
        plt.plot([yROI[0], yROI[1]], [xROI[0], xROI[0]], '-r', 1)
        plt.plot([yROI[0], yROI[1]], [xROI[1], xROI[1]], '-r', 1)

        '''
        按步长生成空间等步长的计算点
        '''
        self.cal_pointsX = np.arange(xROI[0] + half_subset, xROI[1] - half_subset + step, step)
        self.cal_pointsY = np.arange(yROI[0] + half_subset, yROI[1] - half_subset + step, step)
        PMeshX, PMeshY = np.meshgrid(self.cal_pointsX, self.cal_pointsY, indexing='ij')
        # 计算点个数
        self.Lx = len(self.cal_pointsX)
        self.Ly = len(self.cal_pointsY)
        # 计算点坐标
        self.cal_points = np.concatenate(
            (PMeshX.reshape(-1, 1), PMeshY.reshape(-1, 1)), axis=1)

        '''
        初始点
        '''
        if self.ifauto:
            # 中心点被选择为初始点
            x = self.sizeX / 2
            y = self.sizeY / 2
            dist = np.sqrt(np.square(PMeshX.reshape(-1, 1) - x) + np.square(PMeshY.reshape(-1, 1) - y))
            index = np.argmin(dist)
            self.init_point = np.concatenate((PMeshX.reshape(-1, 1)[index], PMeshY.reshape(-1, 1)[index], [1]), axis=0)
        else:
            # 手动选择初始点
            temp = plt.ginput(1)
            x = temp[0][1]
            y = temp[0][0]
            dist = np.sqrt(np.square(PMeshX.reshape(-1, 1) - x) + np.square(PMeshY.reshape(-1, 1) - y))
            index = np.argmin(dist)
            self.init_point = np.concatenate((PMeshX.reshape(-1, 1)[index], PMeshY.reshape(-1, 1)[index], [1]), axis=0)

        plt.plot(self.init_point[1], self.init_point[0], 'r*', 12)
        plt.show()

        self.index_init_point = index
        self.num_cal_points = self.Lx * self.Ly

    def grad_ref_img(self):
        hx = (np.array([1, -8, 0, 8, -1]) / 12).reshape(5, 1)
        hy = (np.array([1, -8, 0, 8, -1]) / 12).reshape(1, 5)
        self.gradx_img = scipy.signal.correlate2d(self.ref_img, hx, 'same')
        self.grady_img = scipy.signal.correlate2d(self.ref_img, hy, 'same')

    def start(self):
        # 归一化
        if self.Normalization:
            self.ref_img = self.ref_img - np.mean(self.ref_img)
            self.ref_img = self.ref_img / (np.max(abs(self.ref_img)))
            self.tar_img = self.tar_img - np.mean(self.tar_img)
            self.tar_img = self.tar_img / (np.max(abs(self.tar_img)))
        # 计算一次参考图像的梯度
        self.grad_ref_img()
        # 计算初始种子点的位移init_disp
        if self.ifauto:
            # 自动的话就假设整像素位移为0
            # init_moved_xy = self.init_point[0:2]
            start = time.perf_counter()
            # init_moved_xy = self.find_point(self.init_point[0:2], 'GA')
            # init_moved_xy = self.find_point(self.init_point[0:2], '逐点搜素')
            init_moved_xy = self.find_point(self.init_point[0:2], '十字搜索')
            # init_moved_xy = self.find_point(self.init_point[0:2], '粗细搜索')
            # init_moved_xy = self.find_point(self.init_point[0:2], '粗细十字搜索')
            # init_moved_xy = self.find_point(self.init_point[0:2], '手动给定')
            end = time.perf_counter()
            # 所用时间
            Duration =(end - start)
        else:
            init_moved_xy = self.find_point(self.init_point[0:2], '逐点搜素')

        self.init_disp = init_moved_xy - self.init_point[0:2]

        if self.sub_pixel_method == 'IC-GN':
            # 一阶形函数情况
            out_points = np.zeros((self.num_cal_points, 6))
            p = np.array([self.init_disp[0], 0, 0, self.init_disp[1], 0, 0]).reshape(-1, 1)
        elif self.sub_pixel_method == 'IC-GN2':
            # 二阶形函数情况
            out_points = np.zeros((self.num_cal_points, 12))
            p = np.array([self.init_disp[0], 0, 0, 0, 0, 0, self.init_disp[1], 0, 0, 0, 0, 0]).reshape(-1, 1)

        # disp, ZNCC, iter_num = self.dic_match(p, out_points)
        disp, ZNCC, iter_num = self.dic_match_reliability_guide(p, out_points)

        return disp, ZNCC, iter_num

    def find_point(self, ref_point, method):
        if method == '逐点搜素':
            subsize = self.subset_size
            ref_point = ref_point.astype('int64')
            sizeX = self.sizeX
            sizeY = self.sizeY
            fSubset = self.ref_img[ref_point[0] - subsize // 2:ref_point[0] + subsize // 2 + 1,
                      ref_point[1] - subsize // 2:ref_point[1] + subsize // 2 + 1]
            deltafVec = fSubset - np.mean(fSubset)
            deltaf = np.sqrt(np.sum(deltafVec ** 2))
            min_Cznssd = 4
            flag = False
            for i in range(sizeX - subsize + 1):
                for j in range(sizeY - subsize + 1):
                    current_gSubset = self.tar_img[i:i + subsize,
                                      j:j + subsize]
                    deltagVec = current_gSubset - np.mean(current_gSubset)
                    deltag = np.sqrt(np.sum(deltagVec ** 2))
                    Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))
                    if Cznssd < 0.005:
                        max_xy = np.array([i + subsize // 2, j + subsize // 2])
                        flag = True
                        break
                    if Cznssd < min_Cznssd:
                        min_Cznssd = Cznssd
                        max_xy = np.array([i + subsize // 2, j + subsize // 2])
                if flag:
                    break

        elif method == '粗细搜索':
            '''分三步，步长分别为4像素、2像素、1像素'''
            subset_size = self.subset_size
            half_subset = int((subset_size - 1) / 2)
            ref_point = ref_point.astype('int64')

            fSubset = ref_img[ref_point[0] - half_subset:ref_point[0] + half_subset + 1,
                      ref_point[1] - half_subset:ref_point[1] + half_subset + 1]
            deltafVec = fSubset - np.mean(fSubset)
            deltaf = np.sqrt(np.sum(deltafVec ** 2))

            sizeX = self.sizeX
            sizeY = self.sizeY
            x = np.arange(0, sizeX - subset_size + 1, 4) + subset_size // 2
            y = np.arange(0, sizeY - subset_size + 1, 4) + subset_size // 2
            X, Y = np.meshgrid(x, y)
            # 所有位移后的点的坐标
            all_xy = np.vstack((X.flatten('F'), Y.flatten('F'))).T.astype('int32')
            Cznssd = np.zeros(len(all_xy))
            for i in range(len(all_xy)):
                current_gSubset = tar_img[all_xy[i, 0] - half_subset:all_xy[i, 0] + half_subset + 1,
                                  all_xy[i, 1] - half_subset:all_xy[i, 1] + half_subset + 1]
                deltagVec = current_gSubset - np.mean(current_gSubset)
                deltag = np.sqrt(np.sum(deltagVec ** 2))
                Cznssd[i] = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))

            xy_1 = all_xy[np.nanargmin(Cznssd)]
            all_xy = xy_1 + np.array([[-2, -2], [-2, 0], [-2, 2], [-0, -2], [-0, 0], [-0, 2], [2, -2], [2, 0], [2, 2]])

            Cznssd = np.zeros(len(all_xy))
            for i in range(len(all_xy)):
                current_gSubset = tar_img[all_xy[i, 0] - half_subset:all_xy[i, 0] + half_subset + 1,
                                  all_xy[i, 1] - half_subset:all_xy[i, 1] + half_subset + 1]
                deltagVec = current_gSubset - np.mean(current_gSubset)
                deltag = np.sqrt(np.sum(deltagVec ** 2))
                Cznssd[i] = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))

            xy_2 = all_xy[np.nanargmin(Cznssd)]
            all_xy = xy_2 + np.array([[-1, -1], [-1, 0], [-1, 1], [-0, -1], [-0, 0], [-0, 1], [1, -1], [1, 0], [1, 1]])

            Cznssd = np.zeros(len(all_xy))
            for i in range(len(all_xy)):
                current_gSubset = tar_img[all_xy[i, 0] - half_subset:all_xy[i, 0] + half_subset + 1,
                                  all_xy[i, 1] - half_subset:all_xy[i, 1] + half_subset + 1]
                deltagVec = current_gSubset - np.mean(current_gSubset)
                deltag = np.sqrt(np.sum(deltagVec ** 2))
                Cznssd[i] = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))

            max_xy = all_xy[np.nanargmin(Cznssd)]

        elif method == '粗细十字搜索':
            subset_size = self.subset_size
            half_subset = int((subset_size - 1) / 2)
            ref_point = ref_point.astype('int64')
            fSubset = ref_img[ref_point[0] - half_subset:ref_point[0] + half_subset + 1,
                      ref_point[1] - half_subset:ref_point[1] + half_subset + 1]
            deltafVec = fSubset - np.mean(fSubset)
            deltaf = np.sqrt(np.sum(deltafVec ** 2))
            sizeX = self.sizeX
            sizeY = self.sizeY
            x = np.arange(0, sizeX - subset_size + 1, 4) + subset_size // 2
            y = np.arange(0, sizeY - subset_size + 1, 4) + subset_size // 2
            X, Y = np.meshgrid(x, y)
            # 所有位移后的点的坐标
            all_xy = np.vstack((X.flatten('F'), Y.flatten('F'))).T.astype('int32')
            Cznssd = np.zeros(len(all_xy))
            for i in range(len(all_xy)):
                current_gSubset = tar_img[all_xy[i, 0] - half_subset:all_xy[i, 0] + half_subset + 1,
                                  all_xy[i, 1] - half_subset:all_xy[i, 1] + half_subset + 1]
                deltagVec = current_gSubset - np.mean(current_gSubset)
                deltag = np.sqrt(np.sum(deltagVec ** 2))
                Cznssd[i] = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))

            xy_1 = all_xy[np.nanargmin(Cznssd)]
            min_Cznssd = 4
            max_x = xy_1[0]
            max_y = xy_1[1]
            old_x = None
            old_y = None
            while True:
                for x in range(7):
                    x = x + max_x-half_subset-3
                    current_gSubset = self.tar_img[x:x + subset_size, max_y - half_subset:max_y + half_subset + 1]
                    deltagVec = current_gSubset - np.mean(current_gSubset)
                    deltag = np.sqrt(np.sum(deltagVec ** 2))
                    Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))
                    if Cznssd < min_Cznssd:
                        min_Cznssd = Cznssd
                        max_x = x + half_subset
                for y in range(7):
                    y = y + max_y-half_subset-3
                    current_gSubset = self.tar_img[max_x - half_subset:max_x + half_subset + 1, y:y + subset_size]
                    deltagVec = current_gSubset - np.mean(current_gSubset)
                    deltag = np.sqrt(np.sum(deltagVec ** 2))
                    Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))
                    if Cznssd < min_Cznssd:
                        min_Cznssd = Cznssd
                        max_y = y + half_subset
                if old_x == max_x and old_y == max_y:
                    break
                old_x = max_x
                old_y = max_y

            max_xy = np.array([max_x, max_y])

        elif method == 'GA':
            '''
            这个方法不是很靠谱
            '''
            sizeX = self.sizeX
            sizeY = self.sizeY
            ref_point = ref_point.astype('int64')
            max_xy = find_point_GA.run(self.ref_img, self.tar_img, self.subset_size, ref_point, sizeX, sizeY)

        elif method == 'GA-cross':
            sizeX = self.sizeX
            sizeY = self.sizeY
            ref_point = ref_point.astype('int64')
            max_xy = find_point_GA.cross_run(self.ref_img, self.tar_img, self.subset_size, ref_point, sizeX, sizeY)

        elif method == '十字搜索':
            subsize = self.subset_size
            half_subset = subsize // 2
            ref_point = ref_point.astype('int64')
            sizeX = self.sizeX
            sizeY = self.sizeY
            fSubset = self.ref_img[ref_point[0] - subsize // 2:ref_point[0] + subsize // 2 + 1,
                      ref_point[1] - subsize // 2:ref_point[1] + subsize // 2 + 1]
            deltafVec = fSubset - np.mean(fSubset)
            deltaf = np.sqrt(np.sum(deltafVec ** 2))

            min_Cznssd = 4
            max_x = ref_point[0]
            max_y = ref_point[1]
            old_x = None
            old_y = None
            while True:
                for x in range(sizeX - subsize + 1):
                    current_gSubset = self.tar_img[x:x + subsize, max_y-half_subset:max_y + half_subset+1]
                    deltagVec = current_gSubset - np.mean(current_gSubset)
                    deltag = np.sqrt(np.sum(deltagVec ** 2))
                    Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))
                    if Cznssd < min_Cznssd:
                        min_Cznssd = Cznssd
                        max_x = x + half_subset
                for y in range(sizeY - subsize + 1):
                    current_gSubset = self.tar_img[max_x-half_subset:max_x + half_subset+1, y:y + subsize]
                    deltagVec = current_gSubset - np.mean(current_gSubset)
                    deltag = np.sqrt(np.sum(deltagVec ** 2))
                    Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))
                    if Cznssd < min_Cznssd:
                        min_Cznssd = Cznssd
                        max_y = y + half_subset
                if old_x == max_x and old_y == max_y:
                    break
                old_x = max_x
                old_y = max_y

            max_xy = np.array([max_x, max_y])

        elif method == '手动给定':
            '''
            会在手动给定的点周围31*31位移范围进行搜索
            '''
            temp = plt.ginput(1)
            y = temp[0][0]
            x = temp[0][1]
            x = int(x)
            y = int(y)
            subsize = self.subset_size
            ref_point = ref_point.astype('int64')
            fSubset = self.ref_img[ref_point[0] - subsize // 2:ref_point[0] + subsize // 2 + 1,
                      ref_point[1] - subsize // 2:ref_point[1] + subsize // 2 + 1]
            deltafVec = fSubset - np.mean(fSubset)
            deltaf = np.sqrt(np.sum(deltafVec ** 2))
            min_Cznssd = 4
            flag = False
            for i in range(51):
                for j in range(51):
                    current_gSubset = self.tar_img[x - 25 + i - subsize // 2:x - 25 + i + subsize // 2 + 1,
                                      y - 25 + j - subsize // 2:y - 25 + j + subsize // 2 + 1]
                    deltagVec = current_gSubset - np.mean(current_gSubset)
                    deltag = np.sqrt(np.sum(deltagVec ** 2))
                    Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))
                    if Cznssd < 0.005:
                        max_xy = np.array([x - 25 + i, y - 25 + j])
                        flag = True
                        break
                    if Cznssd < min_Cznssd:
                        min_Cznssd = Cznssd
                        max_xy = np.array([x - 25 + i, y - 25 + j])
                if flag:
                    break
            plt.plot(max_xy[1], max_xy[0], '+r', 8)
            plt.show()

        return max_xy

    def dic_match(self, p0, out_points):
        Lx = self.Lx
        Ly = self.Ly
        cal_points = self.cal_points
        ZNCC = np.zeros((self.num_cal_points, 1))
        iter_num = np.zeros((self.num_cal_points, 1))
        disp = np.zeros((self.num_cal_points, 2))

        print('共' + str(Lx * Ly) + '个计算点')
        m = 1
        '''
        每个点都以初始位移为0进行遍历迭代优化，没有可靠性引导
        '''
        for i in range(Lx):
            for j in range(Ly):
                point_index = i * Ly + j
                # 跳过初始点去下一个点
                p = p0
                pCoord = np.append(cal_points[point_index, :], 1).reshape(3, 1).astype('int32')
                p, Czncc, Iter, Disp = self.iter_control(pCoord, p)
                ZNCC[point_index, :] = Czncc
                out_points[point_index, :] = p.T
                disp[point_index, :] = Disp
                iter_num[point_index, 0] = Iter
                # 打印进度
                print([m, Lx * Ly])
                m = m + 1

        return disp, ZNCC, iter_num

    # 按可靠性引导的路径进行计算
    def dic_match_reliability_guide(self, p, out_points):
        Lx = self.Lx
        Ly = self.Ly
        cal_points = self.cal_points
        ZNCC = np.zeros((self.num_cal_points, 1))
        iter_num = np.zeros((self.num_cal_points, 1))
        disp = np.zeros((self.num_cal_points, 2))
        # 空队列
        queue = []
        # 用于寻找高可靠性点的四个邻居
        neighbor = np.array([[-1, 0, 1, 0],
                             [0, -1, 0, 1]])
        # m用来控制顺序
        m = 1
        pCoord = self.init_point.reshape(3, 1).astype('int32')

        print('共' + str(Lx * Ly) + '个计算点')
        n = 1

        while queue or m <= 2:
            if m == 1:
                self.thre = 1e-10
                p, Czncc, Iter, Disp = self.iter_control(pCoord, p)
                ZNCC[self.index_init_point, :] = Czncc
                out_points[self.index_init_point, :] = p.T
                disp[self.index_init_point, :] = Disp

                # 保存初始点
                self.tar_ref_init_points = Disp.T  # 不知道这行有啥用
                m = m + 1
                u, v = np.unravel_index(self.index_init_point, [Lx, Ly], 'F')
                queue.append((u, v, Czncc))
                pInit = p
                self.thre = 1e-3

            for neighbor_index in range(4):
                ii = neighbor[0, neighbor_index]
                jj = neighbor[1, neighbor_index]
                i = u + ii
                j = v + jj
                if i < 0 or j < 0 or i > Lx - 1 or j > Ly - 1 or ZNCC[i * Ly + j, 0] != 0:
                    continue
                else:
                    point_index = i * Ly + j
                    # 跳过初始点计算后面的点
                    p = pInit
                    p[0] = p[0] + p[1] * self.step * ii + p[2] * self.step * jj
                    p[3] = p[3] + p[3 + 1] * self.step * ii + p[3 + 2] * self.step * jj

                    pCoord = np.append(cal_points[point_index, :], 1).reshape(3, 1).astype('int32')
                    p, Czncc, Iter, Disp = self.iter_control(pCoord, p)
                    ZNCC[point_index, :] = Czncc
                    out_points[point_index, :] = p.T
                    disp[point_index, :] = Disp
                    iter_num[point_index, 0] = Iter
                    # (i, j)是本次计算点
                    queue.append((i, j, Czncc))
                    m = m + 1

            # 队列queue按照Czncc的大小升序排序，高可靠性的点优先计算邻点
            queue.sort(key=lambda x: x[2])
            u = queue[-1][0]
            v = queue[-1][1]
            pInit = out_points[u * Ly + v].reshape(-1, 1)
            del queue[-1]

            # 打印进度
            print([n, Lx * Ly])
            n = n + 1

        return disp, ZNCC, iter_num

    # 根据参数中的sub_pixel_method进行不同的迭代
    def iter_control(self, pCoord, p):
        if self.sub_pixel_method == 'IC-GN':
            return self.iter_ICGN(pCoord, p)
        elif self.sub_pixel_method == 'IC-GN2':
            return self.iter_ICGN2(pCoord, p)

    # 单个点迭代
    def iter_ICGN(self, pCoord, p):
        Iter = 0
        subset_size = self.subset_size
        localSubHom = self.localSubHom
        deltaVecX = self.deltaVecX
        deltaVecY = self.deltaVecY
        gradx_img = self.gradx_img
        grady_img = self.grady_img
        localSub = self.localSub

        if self.Normalization:
            localSub = localSub / np.tile(np.ceil([subset_size / 2, subset_size / 2]), (len(localSub), 1))
            M = np.diag([1, 1 / subset_size, 1 / subset_size, 1, 1 / subset_size, 1 / subset_size])
        else:
            M = np.eye(6)

        # 估计海森矩阵（Hessian）的deltaf
        nablaf_x = gradx_img[np.ix_(pCoord[0] - 1 + deltaVecX, pCoord[1] - 1 + deltaVecY)]
        nablaf_y = grady_img[np.ix_(pCoord[0] - 1 + deltaVecX, pCoord[1] - 1 + deltaVecY)]
        nablaf = np.concatenate((nablaf_x.reshape((-1, 1), order='F'), nablaf_y.reshape((-1, 1), order='F')), 1)
        J = np.vstack((nablaf[:, 0], localSub[:, 0] * nablaf[:, 0], localSub[:, 1] * nablaf[:, 0], nablaf[:, 1],
                       localSub[:, 0] * nablaf[:, 1], localSub[:, 1] * nablaf[:, 1])).T
        H = J.T @ J
        inv_H = np.linalg.inv(H)
        fSubset = self.ref_img[np.ix_(pCoord[0] - 1 + deltaVecX, pCoord[1] - 1 + deltaVecY)]
        fSubset = fSubset.reshape((-1, 1), order='F')

        deltafVec = fSubset - np.mean(fSubset)
        deltaf = np.sqrt(np.sum(deltafVec ** 2))
        inv_H_J = inv_H @ J.T

        # 由参数向量计算扭曲函数
        warp = self.p_to_wrap(p)

        thre = 1
        # 迭代优化参数向量p
        while thre > 1e-3 and Iter < self.max_iter or Iter == 0:
            # 由扭曲函数求得点在目标图像中的坐标
            gIntep = warp @ localSubHom
            PcoordInt = pCoord + gIntep - np.array([[0], [0], [1]])

            # 所有点仍然位于目标图像内
            if np.prod(PcoordInt[0:2].min(1) > [3, 3]) and np.prod(
                    PcoordInt[0:2].min(1) < [self.sizeX - 3, self.sizeY - 3]):
                # 双三次B样条插值
                tarIntp = self.bicubic_Bspline_interp(self.tar_img, PcoordInt)

                deltagVec = tarIntp - np.mean(tarIntp)
                deltag = np.sqrt(np.sum(deltagVec ** 2))

                delta = deltafVec - deltaf / deltag * deltagVec

                deltap = -inv_H_J @ delta
                deltap = M @ deltap

                # 更新扭曲函数
                deltawarp = self.p_to_wrap(deltap)
                warp = warp @ np.linalg.inv(deltawarp)

                thre = np.sqrt(deltap[0] ** 2 + deltap[3] ** 2)
                Iter = Iter + 1

                # 更新参数向量
                p = np.array([warp[0, 2], warp[0, 0] - 1, warp[0, 1], warp[1, 2], warp[1, 0], warp[1, 1] - 1]).reshape(
                    -1, 1)
                Disp = np.array([warp[0, 2], warp[1, 2]])

                Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))
                Czncc = 1 - 0.5 * Cznssd

            else:
                p = np.zeros((6, 1))
                Czncc = -1
                Iter = Iter + 1
                Disp = np.full([1, 2], np.nan)
                break

        return p, Czncc, Iter, Disp

    # 单个点迭代,二阶形函数
    def iter_ICGN2(self, pCoord, p):
        Iter = 0
        subset_size = self.subset_size
        localSubHom = self.localSubHom
        deltaVecX = self.deltaVecX
        deltaVecY = self.deltaVecY
        gradx_img = self.gradx_img
        grady_img = self.grady_img
        localSub = self.localSub

        if self.Normalization:
            localSub = localSub / np.tile(np.ceil([subset_size / 2, subset_size / 2]), (len(localSub), 1))
            M = np.diag(
                [1, 1 / subset_size, 1 / subset_size, 1 / subset_size ** 2, 1 / subset_size ** 2, 1 / subset_size ** 2,
                 1, 1 / subset_size, 1 / subset_size, 1 / subset_size ** 2, 1 / subset_size ** 2, 1 / subset_size ** 2])
        else:
            M = np.eye(12)

        nablaf_x = gradx_img[np.ix_(pCoord[0] - 1 + deltaVecX, pCoord[1] - 1 + deltaVecY)]
        nablaf_y = grady_img[np.ix_(pCoord[0] - 1 + deltaVecX, pCoord[1] - 1 + deltaVecY)]
        nablaf = np.concatenate((nablaf_x.reshape((-1, 1), order='F'), nablaf_y.reshape((-1, 1), order='F')), 1)
        deltaW2P = np.vstack((np.ones((1, subset_size ** 2)), localSub[:, 0], localSub[:, 1],
                              1 / 2 * localSub[:, 0] ** 2, localSub[:, 0] * localSub[:, 1],
                              1 / 2 * localSub[:, 1] ** 2)).T
        J = np.hstack((np.tile(nablaf[:, 0].reshape(-1, 1), (1, 6)) * deltaW2P,
                       np.tile(nablaf[:, 1].reshape(-1, 1), (1, 6)) * deltaW2P))
        H = J.T @ J
        inv_H = np.linalg.inv(H)
        fSubset = self.ref_img[np.ix_(pCoord[0] - 1 + deltaVecX, pCoord[1] - 1 + deltaVecY)]
        fSubset = fSubset.reshape((-1, 1), order='F')
        deltafVec = fSubset - np.mean(fSubset)
        deltaf = np.sqrt(np.sum(deltafVec ** 2))
        inv_H_J = inv_H @ J.T
        # 由参数向量计算扭曲函数
        warp = self.p_to_wrap(p)
        thre = 1
        # 迭代优化参数向量p
        while thre > 1e-3 and Iter < self.max_iter or Iter == 0:
            # 由扭曲函数求得点在目标图像中的坐标
            gIntep = warp[[3, 4, 5]] @ np.vstack(
                (localSubHom[0] ** 2, localSubHom[0] * localSubHom[1], localSubHom[1] ** 2, localSubHom))
            PcoordInt = pCoord + gIntep - np.array([[0], [0], [1]])
            # 所有点仍然位于目标图像内
            if np.prod(PcoordInt[0:2].min(1) > [3, 3]) and np.prod(
                    PcoordInt[0:2].min(1) < [self.sizeX - 3, self.sizeY - 3]):
                # 双三次B样条插值
                tarIntp = self.bicubic_Bspline_interp(self.tar_img, PcoordInt)
                deltagVec = tarIntp - np.mean(tarIntp)
                deltag = np.sqrt(np.sum(deltagVec ** 2))
                delta = deltafVec - deltaf / deltag * deltagVec
                deltap = -inv_H_J @ delta
                deltap = M @ deltap
                # 更新扭曲函数
                deltawarp = self.p_to_wrap(deltap)
                warp = warp @ np.linalg.inv(deltawarp)

                thre = np.sqrt(deltap[0] ** 2 + deltap[6] ** 2)
                Iter = Iter + 1
                # 更新参数向量
                p = np.array([warp[3, 5], warp[3, 3] - 1, warp[3, 4], 2 * warp[3, 0], warp[3, 1], 2 * warp[3, 2],
                              warp[4, 5], warp[4, 4] - 1, warp[4, 3], 2 * warp[4, 0], warp[4, 1],
                              2 * warp[4, 2]]).reshape(-1, 1)

                Disp = np.array([warp[3, 5], warp[4, 5]])
                Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))
                Czncc = 1 - 0.5 * Cznssd
            else:
                p = np.zeros((12, 1))
                Czncc = -1
                Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2)) / subset_size ** 2
                Iter = Iter + 1
                Disp = np.full([1, 2], np.nan)
                break
        return p, Czncc, Iter, Disp

    def p_to_wrap(self, p):
        if len(p) == 6:
            return np.array([[1 + p[1][0], p[2][0], p[0][0]],
                             [p[4][0], 1 + p[5][0], p[3][0]],
                             [0, 0, 1]])
        else:
            s1 = 2 * p[1][0] + p[1][0] ** 2 + p[0][0] * p[3][0]
            s2 = 2 * p[0][0] * p[4][0] + 2 * (1 + p[1][0]) * p[2][0]
            s3 = p[2][0] ** 2 + p[0][0] * p[5][0]
            s4 = 2 * p[0][0] * (1 + p[1][0])
            s5 = 2 * p[0][0] * p[2][0]
            s6 = p[0][0] ** 2
            s7 = 1 / 2 * (p[6][0] * p[3][0] + 2 * (1 + p[1][0]) * p[7][0] + p[0][0] * p[9][0])
            s8 = p[2][0] * p[7][0] + p[1][0] * p[8][0] + p[6][0] * p[4][0] + p[0][0] * p[10][0] + p[8][0] + p[1][0]
            s9 = 1 / 2 * (p[6][0] * p[5][0] + 2 * (1 + p[8][0]) * p[2][0] + p[0][0] * p[11][0])
            s10 = p[6][0] + p[6][0] * p[1][0] + p[0][0] * p[7][0]
            s11 = p[0][0] + p[6][0] * p[2][0] + p[0][0] * p[8][0]
            s12 = p[0][0] * p[6][0]
            s13 = p[7][0] ** 2 + p[6][0] * p[9][0]
            s14 = 2 * p[6][0] * p[10][0] + 2 * p[7][0] * (1 + p[8][0])
            s15 = 2 * p[8][0] + p[8][0] ** 2 + p[6][0] * p[11][0]
            s16 = 2 * p[6][0] * p[7][0]
            s17 = 2 * p[6][0] * (1 + p[8][0])
            s18 = p[6][0] ** 2

            return np.array([[1 + s1, s2, s3, s4, s5, s6],
                             [s7, 1 + s8, s9, s10, s11, s12],
                             [s13, s14, 1 + s15, s16, s17, s18],
                             [1 / 2 * p[3][0], p[4][0], 1 / 2 * p[5][0], 1 + p[1][0], p[2][0], p[0][0]],
                             [1 / 2 * p[9][0], p[10][0], 1 / 2 * p[11][0], p[7][0], 1 + p[8][0], p[6][0]],
                             [0, 0, 0, 0, 0, 1]]).astype('float64')

    @staticmethod
    def bicubic_Bspline_interp(img, PcoordInt):
        xInt = np.floor(PcoordInt)
        deltaX = PcoordInt - xInt
        numPt = xInt.shape[1]
        MBT = np.array([[-1, 3, -3, 1],
                        [3, -6, 0, 4],
                        [-3, 3, 3, 1],
                        [1, 0, 0, 0]]) / 6
        deltaMatX = MBT @ np.vstack((deltaX[0] ** 3, deltaX[0] ** 2, deltaX[0], np.ones(numPt)))
        deltaMatY = MBT @ np.vstack((deltaX[1] ** 3, deltaX[1] ** 2, deltaX[1], np.ones(numPt)))

        # 参考图像中计算点的索引
        index = np.tile(np.vstack((xInt[1] - 2, xInt[1] - 1, xInt[1], xInt[1] + 1)), (4, 1)) * len(img) + np.vstack(
            (np.tile(xInt[0] - 1, (4, 1)), np.tile(xInt[0], (4, 1)), np.tile(xInt[0] + 1, (4, 1)),
             np.tile(xInt[0] + 2, (4, 1)))) - 1
        D_all = img.flatten('F')[index.astype('int32')]
        tarIntp = np.tile(deltaMatY, (4, 1)) * D_all * np.vstack((np.tile(deltaMatX[0], (4, 1)),
                                                                  np.tile(deltaMatX[1], (4, 1)),
                                                                  np.tile(deltaMatX[2], (4, 1)),
                                                                  np.tile(deltaMatX[3], (4, 1))))

        return np.sum(tarIntp, 0).reshape(-1, 1)


if __name__ == '__main__':
    # ref_img = imread(
    #     'D:\桌面\毕设\别人代码\DIC_ICLM_MATLAB-master\DIC_ICLM_MATLAB-master\Sample Image\Int translation\img_00000.bmp', '0')
    # tar_img = imread(
    #     'D:\桌面\毕设\别人代码\DIC_ICLM_MATLAB-master\DIC_ICLM_MATLAB-master\Sample Image\Int translation\img_00303.bmp', '0')
    # ref_img = imread(
    #     'D:/桌面/毕设/pictures/3_0.bmp', '0')
    # ref_img = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
    # tar_img = imread(
    #     'D:/桌面/毕设/pictures/3_1.bmp', '0')
    # tar_img = cv.cvtColor(tar_img, cv.COLOR_BGR2GRAY)

    ref_img_dict = ['data/x y方向位移虚拟散斑/15.23pix-y方向位移/ImgSpeck1.bmp',
                    'data/x y方向位移虚拟散斑/18.23pix-y方向 15.8pix-x方向/ImgSpeck1.bmp']
    tar_img_dict = ['data/x y方向位移虚拟散斑/15.23pix-y方向位移/ImgSpeck2.bmp',
                    'data/x y方向位移虚拟散斑/18.23pix-y方向 15.8pix-x方向/ImgSpeck2.bmp']

    ref_img = imread(ref_img_dict[1], '0')
    tar_img = imread(tar_img_dict[1], '0')

    dic = DIC(ref_img, tar_img)
    dic.set_parameters()

    plt.subplot(221)
    plt.title('reference img')
    plt.imshow(ref_img, cmap='gray')

    # 在参考图像上选择计算点
    dic.calculate_points()

    plt.subplot(222)
    plt.title('target img')
    plt.imshow(tar_img, cmap='gray')

    result = dic.start()

    # 绘制结果图像
    xy = result[0]
    x = xy[:, 0]
    x = x.reshape(dic.Lx, dic.Ly)
    y = xy[:, 1]
    y = y.reshape(dic.Lx, dic.Ly)
    temp = 5  # 控制刻度
    plt.subplot(223)
    plt.title('X displacement')
    plt.imshow(x, cmap='nipy_spectral')
    plt.colorbar()
    plt.xticks(np.arange(0, dic.Ly, temp), np.arange(0, dic.Ly * dic.step, temp * dic.step))
    plt.yticks(np.arange(0, dic.Lx, temp), np.arange(0, dic.Lx * dic.step, temp * dic.step))

    plt.subplot(224)
    plt.title('Y displacement')
    plt.imshow(y, cmap='nipy_spectral')
    plt.colorbar()
    plt.xticks(np.arange(0, dic.Ly, temp), np.arange(0, dic.Ly * dic.step, temp * dic.step))
    plt.yticks(np.arange(0, dic.Lx, temp), np.arange(0, dic.Lx * dic.step, temp * dic.step))
