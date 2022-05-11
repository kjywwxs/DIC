import cv2 as cv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math


# digits为需要转换成的二进制位数，一个坐标转2*digits位
def encode(group_xy, digits):
    binary_repr_v = np.vectorize(np.binary_repr)
    temp = binary_repr_v(group_xy, digits)
    group_01 = np.zeros((len(group_xy), 2 * digits))
    for i in range(len(temp)):
        group_01[i] = np.array([int(x) for x in temp[i, 0] + temp[i, 1]])
    return group_01


def decode(group_01, digits):
    resdivide = np.split(group_01, 2, axis=1)
    x = np.zeros((len(group_01), 1)).astype('int64')
    y = np.zeros((len(group_01), 1)).astype('int64')
    for i in range(len(group_01)):
        x[i] = resdivide[0][i, :].dot(2 ** np.arange(digits)[::-1])
        y[i] = resdivide[1][i, :].dot(2 ** np.arange(digits)[::-1])

    return np.hstack((x, y))


def run(ref_img, tar_img, subsize, ref_point, sizeX, sizeY):
    fSubset = ref_img[(ref_point[0] - subsize // 2):(ref_point[0] + subsize // 2 + 1),
              (ref_point[1] - subsize // 2):(ref_point[1] + subsize // 2 + 1)]
    deltafVec = fSubset - np.mean(fSubset)
    deltaf = np.sqrt(np.sum(deltafVec ** 2))
    '''
           创建初始种群
           初始种群数量50
           交叉概率0.8
           变异概率0.05
           预设迭代最大次数500次
    '''
    N = 50
    Pc = 0.8
    Pm = 0.05
    M = 500

    x = np.random.randint(sizeX - subsize + 1, size=N) + subsize // 2
    y = np.random.randint(sizeY - subsize + 1, size=N) + subsize // 2
    group_xy = np.vstack((x, y)).T.astype('int32')
    # 一个坐标所需二级制位数的一半
    x_01_len = len(bin(np.max(x))) - 2
    y_01_len = len(bin(np.max(y))) - 2
    digits = [y_01_len, x_01_len][x_01_len > y_01_len]
    max_fitness = 0
    for m in range(M):
        '''
        适应度计算
        有时group_xy坐标会小于15,所以消灭x,y不在50~250之间的个体
        '''
        fitness = np.zeros(len(group_xy))
        for i in range(len(group_xy)):
            if group_xy[i, 0] < subsize // 2 or group_xy[i, 0] > subsize // 2 + sizeX - subsize or group_xy[
                i, 1] < subsize // 2 or group_xy[i, 1] > subsize // 2 + sizeY - subsize:
                # 消灭x, y不在允许范围内的个体
                fitness[i] = 0
            else:
                current_gSubset = tar_img[group_xy[i, 0] - subsize // 2:group_xy[i, 0] + subsize // 2 + 1,
                                  group_xy[i, 1] - subsize // 2:group_xy[i, 1] + subsize // 2 + 1]
                deltagVec = current_gSubset - np.mean(current_gSubset)
                deltag = np.sqrt(np.sum(deltagVec ** 2))
                Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))
                fitness[i] = (4 - Cznssd)

        fitness = np.nan_to_num(fitness)  # 有时候位移距离大过图像子区，会除以0导致nan
        if np.max(fitness) > max_fitness:
            max_fitness = np.max(fitness)
            max_xy = group_xy[np.argmax(fitness)]
        else:
            # 新群体中最高适应度小于上一代最高适应度，用上一代最好替换掉这一代最差
            group_xy[np.argmin(fitness)] = max_xy

        if max_fitness > 3.8:
            break
        '''
        轮盘赌原则进行选择 
        '''
        print(max_fitness)
        p = fitness / sum(fitness)
        # 从0到N-1按概率p取N个数
        choiced_index = np.random.choice(len(group_xy), N, replace=True, p=p)
        new_group_xy = group_xy[choiced_index]
        '''
        交叉 
        N必须为偶数
        '''
        new_group_01 = encode(new_group_xy, digits).astype('int8')  # 编码
        for i in range(0, N, 2):
            pc = np.random.rand()
            if pc < Pc:
                q = np.random.randint(2, size=digits * 2)
                for j in range(digits * 2):
                    if q[j] == 1:
                        temp = new_group_01[i + 1, j]
                        new_group_01[i + 1, j] = new_group_01[i, j]
                        new_group_01[i, j] = temp
        '''
        变异
        '''
        bool_matrix = np.random.rand(N, digits * 2) < Pm
        new_group_01 = np.logical_xor(bool_matrix, new_group_01).astype('int8')

        group_xy = decode(new_group_01, digits)

    return max_xy


# 十字搜索结合遗传算法
def cross_run(ref_img, tar_img, subsize, ref_point, sizeX, sizeY):
    fSubset = ref_img[(ref_point[0] - subsize // 2):(ref_point[0] + subsize // 2 + 1),
              (ref_point[1] - subsize // 2):(ref_point[1] + subsize // 2 + 1)]
    deltafVec = fSubset - np.mean(fSubset)
    deltaf = np.sqrt(np.sum(deltafVec ** 2))
    '''
           创建初始种群
           初始种群数量50
           交叉概率0.8
           变异概率0.05
           预设迭代最大次数500次
    '''
    N = 50
    Pc = 0.8
    Pm = 0.05
    M = 500

    x = np.random.randint(sizeX - subsize + 1, size=N) + subsize // 2
    y = np.random.randint(sizeY - subsize + 1, size=N) + subsize // 2
    group_xy = np.vstack((x, y)).T.astype('int32')
    # 一个坐标所需二级制位数的一半
    x_01_len = len(bin(np.max(x))) - 2
    y_01_len = len(bin(np.max(y))) - 2
    digits = [y_01_len, x_01_len][x_01_len > y_01_len]
    max_fitness = 0
    for m in range(M):
        '''
        适应度计算
        有时group_xy坐标会小于15,所以消灭x,y不在50~250之间的个体
        '''
        fitness = np.zeros(len(group_xy))
        for i in range(len(group_xy)):
            if group_xy[i, 0] < subsize // 2 or group_xy[i, 0] > subsize // 2 + sizeX - subsize or group_xy[
                i, 1] < subsize // 2 or group_xy[i, 1] > subsize // 2 + sizeY - subsize:
                # 消灭x, y不在允许范围内的个体
                fitness[i] = 0
            else:
                current_gSubset = tar_img[group_xy[i, 0] - subsize // 2:group_xy[i, 0] + subsize // 2 + 1,
                                  group_xy[i, 1] - subsize // 2:group_xy[i, 1] + subsize // 2 + 1]
                deltagVec = current_gSubset - np.mean(current_gSubset)
                deltag = np.sqrt(np.sum(deltagVec ** 2))
                Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))
                fitness[i] = (4 - Cznssd)

        fitness = np.nan_to_num(fitness)  # 有时候位移距离大过图像子区，会除以0导致nan
        if np.max(fitness) > max_fitness:
            max_fitness = np.max(fitness)
            max_xy = group_xy[np.argmax(fitness)]
        else:
            # 新群体中最高适应度小于上一代最高适应度，用上一代最好替换掉这一代最差
            group_xy[np.argmin(fitness)] = max_xy

        if max_fitness > 3:
            break
        '''
        轮盘赌原则进行选择 
        '''
        print(max_fitness)
        p = fitness / sum(fitness)
        # 从0到N-1按概率p取N个数
        choiced_index = np.random.choice(len(group_xy), N, replace=True, p=p)
        new_group_xy = group_xy[choiced_index]
        '''
        交叉 
        N必须为偶数
        '''
        new_group_01 = encode(new_group_xy, digits).astype('int8')  # 编码
        for i in range(0, N, 2):
            pc = np.random.rand()
            if pc < Pc:
                q = np.random.randint(2, size=digits * 2)
                for j in range(digits * 2):
                    if q[j] == 1:
                        temp = new_group_01[i + 1, j]
                        new_group_01[i + 1, j] = new_group_01[i, j]
                        new_group_01[i, j] = temp
        '''
        变异
        '''
        bool_matrix = np.random.rand(N, digits * 2) < Pm
        new_group_01 = np.logical_xor(bool_matrix, new_group_01).astype('int8')

        group_xy = decode(new_group_01, digits)
    '''十字搜索'''
    min_Cznssd = 4
    max_x = max_xy[0]
    max_y = max_xy[1]
    half_subset = subsize // 2
    old_x = None
    old_y = None
    while True:
        for x in range(sizeX - subsize + 1):
            current_gSubset = tar_img[x:x + subsize, max_y - half_subset:max_y + half_subset + 1]
            deltagVec = current_gSubset - np.mean(current_gSubset)
            deltag = np.sqrt(np.sum(deltagVec ** 2))
            Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))
            if Cznssd < min_Cznssd:
                min_Cznssd = Cznssd
                max_x = x + half_subset
        for y in range(sizeY - subsize + 1):
            current_gSubset = tar_img[max_x - half_subset:max_x + half_subset + 1, y:y + subsize]
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

    new_max_xy = np.array([max_x, max_y])
    return new_max_xy


if __name__ == '__main__':

    ref_img = mpimg.imread(
        'D:/桌面/毕设/pictures/3_0.bmp', '0')
    ref_img = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
    tar_img = mpimg.imread(
        'D:/桌面/毕设/pictures/3_1.bmp', '0')
    tar_img = cv.cvtColor(tar_img, cv.COLOR_BGR2GRAY)

    focus_point = np.array([100, 120, 1])
    subsize = 31
    fSubset = ref_img[100 - 15:100 + 15 + 1, 120 - 15:120 + 15 + 1]

    '''
           创建初始种群
           种群范围已经给定[50,250]，后期需要修改
           初始种群数量50
           交叉概率0.8
           变异概率0.01
           预设迭代最大次数500次
       '''
    N = 50
    Pc = 0.8
    # Pc_max = 0.9
    # Pc_min = 0.6
    Pm = 0.01
    # Pm_max = 0.1
    # Pm_min = 0.01
    M = 500

    x = np.random.randint(200, size=N) + 50
    y = np.random.randint(200, size=N) + 50
    group_xy = np.vstack((x, y)).T.astype('int32')

    deltafVec = fSubset - np.mean(fSubset)
    deltaf = np.sqrt(np.sum(deltafVec ** 2))

    # 绘图相关
    plt_max_fit = np.zeros(M)
    plt_ave_fit = np.zeros(M)

    max_fitness = 0
    for m in range(M):
        '''
        适应度计算
        有时group_xy坐标会小于15,所以消灭x,y不在50~250之间的个体
        '''
        fitness = np.zeros(len(group_xy))
        for i in range(len(group_xy)):
            if group_xy[i, 0] < 50 or group_xy[i, 0] > 250 or group_xy[i, 1] < 50 or group_xy[i, 1] > 250:
                # 有时group_xy坐标会小于15, 所以消灭x, y不在50 ~250之间的个体
                fitness[i] = 0
            else:
                current_gSubset = tar_img[group_xy[i, 0] - 15:group_xy[i, 0] + 15 + 1,
                                  group_xy[i, 1] - 15:group_xy[i, 1] + 15 + 1]
                deltagVec = current_gSubset - np.mean(current_gSubset)
                deltag = np.sqrt(np.sum(deltagVec ** 2))
                Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))
                fitness[i] = (4 - Cznssd)

        if np.max(fitness) > max_fitness:
            max_fitness = np.max(fitness)
            max_xy = group_xy[np.argmax(fitness)]
        else:
            # 新群体中最高适应度小于上一代最高适应度，用上一代最好替换掉这一代最差
            group_xy[np.argmin(fitness)] = max_xy
            fitness[np.argmin(fitness)] = max_fitness

        # if max_fitness > 3.995:
        #     break
        '''
        轮盘赌原则进行选择
        '''
        print(sum(fitness) / len(group_xy))
        # 绘图相关
        plt_max_fit[m] = max_fitness
        plt_ave_fit[m] = sum(fitness) / len(group_xy)

        p = fitness / sum(fitness)
        # 从0到N-1按概率p取N个数
        choiced_index = np.random.choice(len(group_xy), N, replace=True, p=p)
        new_group_xy = group_xy[choiced_index]
        '''
        用来自适应交叉概率
        '''
        new_group_fit = fitness[choiced_index]
        new_ave_fit = sum(fitness) / N
        '''
        均匀交叉
        N必须为偶数
        '''
        new_group_01 = encode(new_group_xy, 8).astype('int8')  # 编码
        for i in range(0, N, 2):
            pc = np.random.rand()

            # big_fit = [new_group_fit[i+1], new_group_fit[i]][new_group_fit[i] > new_group_fit[i+1]]
            # if big_fit < new_ave_fit:
            #     # 如果适应度小，则交叉概率高
            #     Pc = Pc_max
            # else:
            #     # 如果适应度大，则交叉概率低
            #     Pc = Pc_max-(Pc_max-Pc_min)*(big_fit - new_ave_fit)/(np.max(new_group_fit) - new_ave_fit)
            # if big_fit < new_ave_fit:
            #     Pc = 1
            # else:
            #     Pc = 1*(np.max(new_group_fit)-big_fit)/(np.max(new_group_fit)-new_ave_fit)

            if pc < Pc:
                q = np.random.randint(2, size=16)
                for j in range(16):
                    if q[j] == 1:
                        temp = new_group_01[i + 1, j]
                        new_group_01[i + 1, j] = new_group_01[i, j]
                        new_group_01[i, j] = temp

        # '''
        # 用来自适应变异概率，再算一遍fitness????????????????
        # '''
        # jiaochaed_group = decode(new_group_01, 8)
        # jiaochaed_fitness = np.zeros(len(jiaochaed_group))
        # for i in range(len(jiaochaed_group)):
        #     if jiaochaed_group[i, 0] < 50 or jiaochaed_group[i, 0] > 250 or jiaochaed_group[i, 1] < 50 or \
        #             jiaochaed_group[i, 1] > 250:
        #         # 有时jiaochaed_group坐标会小于15, 所以消灭x, y不在50 ~250之间的个体
        #         jiaochaed_fitness[i] = 0
        #     else:
        #         current_gSubset = tar_img[jiaochaed_group[i, 0] - 15:jiaochaed_group[i, 0] + 15 + 1,
        #                           jiaochaed_group[i, 1] - 15:jiaochaed_group[i, 1] + 15 + 1]
        #         deltagVec = current_gSubset - np.mean(current_gSubset)
        #         deltag = np.sqrt(np.sum(deltagVec ** 2))
        #         Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))
        #         jiaochaed_fitness[i] = (4 - Cznssd)
        # exist = (jiaochaed_fitness != 0)
        # jiaochaed_ave_fit = sum(jiaochaed_fitness) / sum(exist)
        '''
        变异
        '''
        bool_matrix = np.random.rand(N, 16) < Pm
        # temp = np.random.rand(N, 16)
        # bool_matrix = np.zeros((N, 16))
        # for i in range(N):
        #     if jiaochaed_fitness[i] < jiaochaed_ave_fit:
        #         # 适应度低则变异概率高
        #         Pm = Pm_max
        #     else:
        #         Pm = Pm_max - (Pm_max - Pm_min) * (jiaochaed_fitness[i] - jiaochaed_ave_fit) / (
        #                     np.max(jiaochaed_fitness) - jiaochaed_ave_fit)
        #     # if jiaochaed_fitness[i] < jiaochaed_ave_fit:
        #     #     Pm = 0.5
        #     # else:
        #     #     Pm = 0.5 * (np.max(jiaochaed_fitness)-jiaochaed_fitness[i]) / (np.max(jiaochaed_fitness) - jiaochaed_ave_fit)
        #
        #     bool_matrix[i] = temp[i, :] < Pm
        new_group_01 = np.logical_xor(bool_matrix, new_group_01).astype('int8')

        group_xy = decode(new_group_01, 8)
        # print(max_xy)

    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
    M_X = np.arange(M)
    plt.figure(1)
    plt.plot(M_X, plt_max_fit)
    plt.plot(M_X, plt_ave_fit)
    plt.xlabel('迭代次数')
    plt.ylabel('适应度')
    plt.show()
