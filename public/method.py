# # 这是转角双层石墨烯的连续模型和紧束缚模型的代码整理，里面包含了一些功能实现，如吸收谱，拉曼光谱等。
import functools
import numpy as np
import cmath
import multiprocessing
import os.path
import time
from multiprocessing import Pool

import numpy as np
from numpy import exp, sin, array, conj, linspace, real, arange, dot, save, zeros, imag, ones, average, sqrt, pi, cos
# basic parameters
from numpy.linalg import norm, eig
import cv2 as cv
import shutil
import matplotlib

matplotlib.use("agg")
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objs as go
from scipy.linalg import expm
from cmath import log
import plotly.express as px


# # 这部分为在计算当中需要用到的常数
from public.consts import *


class PubMeth:
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        cores_num = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        cores_num = multiprocessing.cpu_count()
    print("Cores Num: ", cores_num)

    @staticmethod  # # 寻找两个数的最大公约数
    def gcd(a, b):
        if a < b:
            return PubMeth.gcd(b, a)
        while a % b != 0:
            temp = b
            b = a % b
            a = temp
        return b

    @staticmethod  # # 寻找质数对
    def get_coprime(limit=5):
        out_coprimes = []
        for r in range(1, limit):
            for m in range(1, limit):
                if PubMeth.gcd(m, r) == 1:
                    out_coprimes.append((m, r))
        return out_coprimes

    @staticmethod  # 在多线程运算当中需要用到的重载函数
    def overdrive_func(
            core_i,
            input_func,
            list_to_count,
            other_args_list,
            trans_out_list, hint=False):
        if hint:
            print("Core %s" % core_i, " is running")
        out_list = [input_func(ele_to_count, *other_args_list)
                    for ele_to_count in list_to_count]
        out_list.append(core_i)
        trans_out_list.append(out_list)

    @staticmethod  # # 这里是多线程运算需要用到的函数，可对计算函数进行重载并进行多进程运算（一般用于对K空间的积分），该函数可极大地利用电脑的核数进行计算
    def multi_proc_func(input_func, divided_list, args_list_f, hint=False):
        out_list_f = multiprocessing.Manager().list()
        args_list = [args_list_f, out_list_f, hint]
        path_num = len(divided_list)
        p_f = Pool(path_num)
        for i in range(path_num):
            tmp_list = args_list[:]
            tmp_list.insert(0, i)  # 进程序号
            tmp_list.insert(1, input_func)  # 输入的函数
            tmp_list.insert(2, divided_list[i])  # 输入的需要计算的（第一个）参数
            p_f.apply_async(PubMeth.overdrive_func, tuple(tmp_list))
        if hint:
            print('Waiting for all subprocesses done...')
        p_f.close()
        p_f.join()
        if hint:
            print('All subprocesses done.')
        total_list = []
        for path_i in range(path_num):
            for ele_list in out_list_f:
                if ele_list[-1] == path_i:
                    total_list.extend(ele_list[0:-1])
        return total_list

    @classmethod  # # 将K空间的k点做一个划分，以便进行多进程运算
    def divide_list(cls, input_list):
        kp_part_list_f = []
        for i in range(cls.cores_num - 1):
            kp_part_list_f.append(input_list[int(len(
                input_list) / cls.cores_num) * i:int(len(input_list) / cls.cores_num) * (i + 1)])
        kp_part_list_f.append(
            input_list[int(len(input_list) / cls.cores_num) * (cls.cores_num - 1):])
        return kp_part_list_f

    @staticmethod  # # 对不同的操作系统可获得正确的保存路径
    def get_right_save_path(input_dir_name):  # only take one parameter
        return os.getcwd() + os.sep + input_dir_name + os.sep

    @staticmethod  # # 单层石墨烯的哈密顿量，已对摩尔布里渊区进行了约化
    def get_k_mat(a0, theta, Kg, kx, ky):
        return array([
            [1 / Kg, sqrt(3) / 2 * a0 * (kx - 1j * ky) * exp(-1j * theta / 2)],
            [sqrt(3) / 2 * a0 * (kx + 1j * ky) * exp(1j * theta / 2), 1 / Kg]
        ])

    @staticmethod  # # 将参数列表转化为矩阵形式
    def list2mat(input_list):
        return array(input_list).reshape(
            (int(sqrt(len(input_list))), int(sqrt(len(input_list)))))

    @staticmethod  # # 将参数列表转化为矩阵形式
    def putin2mat(args_list_in):
        total_list = []
        for ele_args in args_list_in:
            # the second parameter should be an angle(degree)
            total_list.append(ele_args[0] * exp(1j * ele_args[1] / 180 * pi))
        return PubMeth.list2mat(total_list)

    @staticmethod  # # 判断目标点是否位于选定区域内
    def isInterArea(testPoint, AreaPoint):  # testPoint为待测点[x,y]
        # AreaPoint为按顺时针顺序的4个点[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        LBPoint = AreaPoint[0]
        LTPoint = AreaPoint[1]
        RTPoint = AreaPoint[2]
        RBPoint = AreaPoint[3]
        a_f = (LTPoint[0] - LBPoint[0]) * (testPoint[1] - LBPoint[1]) - \
              (LTPoint[1] - LBPoint[1]) * (testPoint[0] - LBPoint[0])
        b_f = (RTPoint[0] - LTPoint[0]) * (testPoint[1] - LTPoint[1]) - \
              (RTPoint[1] - LTPoint[1]) * (testPoint[0] - LTPoint[0])
        c_f = (RBPoint[0] - RTPoint[0]) * (testPoint[1] - RTPoint[1]) - \
              (RBPoint[1] - RTPoint[1]) * (testPoint[0] - RTPoint[0])
        d_f = (LBPoint[0] - RBPoint[0]) * (testPoint[1] - RBPoint[1]) - \
              (LBPoint[1] - RBPoint[1]) * (testPoint[0] - RBPoint[0])
        if (a_f > 0 and b_f > 0 and c_f > 0 and d_f > 0) or (
                a_f < 0 and b_f < 0 and c_f < 0 and d_f < 0):
            return True
        else:
            return False

    @staticmethod  # # 判断点是否位于边界
    def at_corners(chosen_p, corner_list):
        test_p_f = array([chosen_p[0], chosen_p[1]])
        distance_list = []
        for ele_corner in corner_list:
            corner_arr = array([ele_corner[0], ele_corner[1]])
            diff = corner_arr - test_p_f
            distance_list.append(np.linalg.norm(diff))
        min_d = min(distance_list)
        if min_d < 0.001:
            return True

    @staticmethod  # # 判断一个晶格点在哪个超晶格基矢下的映射，其长度最短（用于紧束缚模型）
    def get_smallest_distance(original_distance_arr, input_vec_list):
        transformed_vecs_list = original_distance_arr + array(input_vec_list)
        transformed_norm_list = norm(transformed_vecs_list, axis=1)
        index_min = list(transformed_norm_list).index(
            min(transformed_norm_list))
        return transformed_vecs_list[index_min]

    @staticmethod  # # 判断矢量是否位于矢量列表中
    def arr_index_in_arr_list(vec, vec_list, precision=0.01):
        transformed_vecs_list = vec - array(vec_list)
        transformed_norm_list = norm(transformed_vecs_list, axis=1)
        if min(transformed_norm_list) < precision:
            return list(transformed_norm_list).index(min(transformed_norm_list))
        else:
            return "The vector is not in the list!"

    @staticmethod
    def arr_plus_arr_list(arr, arr_list):  # # returns a list
        return list(arr + arr_list)

    @staticmethod  # # 旋转矩阵，可作用于矢量得到旋转之后的矢量
    def rotation(angle):  # arc unit
        theta = angle / 180 * pi
        rot_mat = array([
            [cos(theta), -sin(theta)],
            [sin(theta), cos(theta)]
        ])
        return rot_mat

    @staticmethod  # # 点到点设定若干个点进行k空间路径描述
    def p2p(point1, point2, density_per_path):
        k_along = []
        if point2[0] != point1[0]:
            k_slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
            number_of_points = int(
                density_per_path * sqrt((point2[1] - point1[1]) ** 2 + (point2[0] - point1[0]) ** 2))
            for xx in linspace(point1[0], point2[0], number_of_points):
                k_along.append((xx, k_slope * (xx - point2[0]) + point2[1]))
        elif point2[0] == point1[0]:
            number_of_points = int(
                density_per_path * sqrt((point2[1] - point1[1]) ** 2))
            for yy in linspace(point1[1], point2[1], number_of_points):
                k_along.append((point2[0], yy))
        return k_along

    @staticmethod  # # 三角格子划分，得到若干个shell下的所有点的矢量列表
    def tri_lattice(shell_i, a0_lattice, rotation_angle=0, shift_arr=array([0, 0])):
        base_vecs = [a0_lattice * array([1, 0]), a0_lattice * array([-1, 0]), a0_lattice * array([1 / 2, sqrt(3) / 2]), a0_lattice * array([1 / 2, -sqrt(3) / 2]), a0_lattice * array([-1 / 2, sqrt(3) / 2]), a0_lattice * array([-1 / 2, -sqrt(3) / 2])]
        all_vecs = []
        ru_list = []
        for i1 in range(1, shell_i + 1):
            all_vecs.extend(list(array(base_vecs) * i1))
        for j in range(1, shell_i):
            tmp_R6 = j * a0_lattice * array([1 / 2, sqrt(3) / 2])
            tmp_R1 = a0_lattice * array([1, 0])
            for i2 in range(1, shell_i + 1 - j):
                ru_list.append(tmp_R6 + i2 * tmp_R1)
        all_vecs.extend(ru_list)
        for i3 in range(1, 6):
            tmp_theta = 60 * i3
            tmp_new = [PubMeth.rotation(tmp_theta)@vec for vec in ru_list]
            all_vecs.extend(tmp_new)
        all_vecs.append(array([0, 0]))
        all_vecs = [PubMeth.rotation(rotation_angle) @ ele_vec for ele_vec in all_vecs]
        all_vecs = array(all_vecs) + shift_arr
        return list(all_vecs)

    @staticmethod
    def rect2diam(input_mat, file_name, title_name, save_2d_plots=False, rm_raw=True, cmap='jet', direction_up=True, save_in_case_same_name=False, save_mat=False):
        # plot of real part
        plt.figure(dpi=300, figsize=(7, 7))
        plt.imshow(input_mat, cmap=cmap, aspect='auto')  # origin='lower')
        plt.axis('off')
        target_dir_raw = PubMeth.get_right_save_path("Pics_raw")
        if save_2d_plots:
            if not os.path.exists(target_dir_raw):
                os.mkdir(target_dir_raw)
            if save_in_case_same_name:
                im_raw_save_func = functools.partial(plt.savefig, bbox_inches='tight', pad_inches=0, dpi=300)
                PubMeth.save_if_same_name(im_raw_save_func, save_name=target_dir_raw + file_name + "_raw_.png", target_dir='')
            else:
                plt.savefig(target_dir_raw + file_name + "_raw_.png", bbox_inches='tight',    pad_inches=0, dpi=300)
        plt.close()

        tmp_imag = cv.imread(target_dir_raw + file_name + "_raw_.png")
        cols, rows = tmp_imag.shape[:2]   # 取长宽
        if direction_up:
            point1 = np.float32([[0, 0], [rows, 0], [0, cols]])
            point2 = np.float32([[int(rows / 2), 0], [rows, int(sqrt(3) / 2 * rows)], [0, int(sqrt(3) / 2 * rows)]])
            M = cv.getAffineTransform(point1, point2)
            dst = cv.warpAffine(tmp_imag, M, (rows, int(sqrt(3) * rows)), borderValue=(255, 255, 255))
        else:
            point1 = np.float32([[0, 0], [rows, 0], [0, cols]])
            point2 = np.float32([[int(cols / 2 * sqrt(3)), 0], [int(cols * sqrt(3)), cols // 2], [0, cols // 2]])
            M = cv.getAffineTransform(point1, point2)
            dst = cv.warpAffine(tmp_imag, M, (int(cols * sqrt(3)), cols), borderValue=(255, 255, 255))
        plt.imshow(dst)  # , origin='lower'  This origin will determine the direction of image
        plt.title(title_name)
        plt.axis('off')
        target_dir_mod = PubMeth.get_right_save_path("Pics_mod")
        if not os.path.exists(target_dir_mod):
            os.mkdir(target_dir_mod)
        if save_in_case_same_name:
            im_mod_save_func = functools.partial(plt.savefig, bbox_inches='tight', pad_inches=0, dpi=300)
            PubMeth.save_if_same_name(im_mod_save_func, target_dir_mod + file_name + "_mod_.png", target_dir='')
        else:
            plt.savefig(target_dir_mod + file_name + "_mod_.png", bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        if rm_raw:
            shutil.rmtree(target_dir_raw)
        target_dir_mat = PubMeth.get_right_save_path('Mat_files')
        if not os.path.exists(target_dir_mat):
            os.mkdir(target_dir_mat)
        if save_mat:
            if save_in_case_same_name:
                mat_save_func = functools.partial(np.save, arr=input_mat)
                PubMeth.save_if_same_name(mat_save_func, save_name=target_dir_mat + file_name + "_mat_.npy", target_dir='')
            else:
                np.save(target_dir_mat + file_name + "_mat_.npy", input_mat)
    
    @staticmethod
    def draw_box(input_vex_arr):
        x_list = array(input_vex_arr)[:, 0]
        y_list = array(input_vex_arr)[:, 1]
        plt.plot(x_list, y_list)

    @staticmethod
    def draw_dots(dots_arr_in, marker_type, save_or_not=False, hold_on=False, draw_width=1, fig_title='', x_label='', y_label='', leg_list=[]):
        x_list = array(dots_arr_in)[:, 0]
        y_list = array(dots_arr_in)[:, 1]

        plt.scatter(x_list, y_list, marker=marker_type, linewidths=draw_width)
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_title(fig_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(leg_list)  # this is usually put at the last operation
        if save_or_not:
            tmp_dir = PubMeth.get_right_save_path("tmp_figs")
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)
            plt.savefig(tmp_dir + "dots.png", dpi=300)
        if not hold_on:
            plt.close()

    @staticmethod
    def commensurate_angle(m0, r):
        value_cos = (3 * m0 ** 2 + 3 * m0 * r + r ** 2 / 2) / (3 * m0 ** 2 + 3 * m0 * r + r ** 2)
        twist_angle = np.arccos(value_cos) / pi * 180
        theta = np.arccos(value_cos)
        return twist_angle, theta

    @staticmethod
    def overlap_points(arr_list1, arr_list2):
        overlap_points = []
        for ele_arr in arr_list1:
            tmp_arr_list = arr_list2 - ele_arr
            norm_list = norm(tmp_arr_list, axis=1)
            sift_list = norm_list <= 0.01
            fit_arrs_list = array(arr_list2)[sift_list]
            overlap_points.append(fit_arrs_list)
        return list(np.unique(np.vstack(tuple(overlap_points)), axis=0))

    @staticmethod
    def plus_set(arr_list1, arr_list2, length_limit=10):
        all_overlap_list = []
        for ele_arr in arr_list1:
            tmp_arr_list = arr_list2 + ele_arr
            all_overlap_list.extend(list(tmp_arr_list))
        second_list = np.unique(array(all_overlap_list), axis=0)
        out_list = second_list[norm(second_list, axis=1) < length_limit]
        return list(out_list)

    @staticmethod
    def unit_cell_per_supercell(m0, r):
        if r % 3 != 0:
            N_unit_cell = int(3 * m0 ** 2 + 3 * m0 * r + r ** 2)
        else:
            N_unit_cell = int(m0 ** 2 + m0 * r + r ** 2 / 3)
        return N_unit_cell

    @staticmethod
    def moire_pattern(m0, r, shell_i=5, a0_constant=1, moire_figsave=False, k_valley_save=False, moire_pm_set_save=False, k_space=True, k_pm_valley_save=False):
        if k_space:
            N = PubMeth.unit_cell_per_supercell(m0, r)
            print("Number of unit cell in one supercell: ", N)

            comm_ang, comm_theta = PubMeth.commensurate_angle(m0, r)
            print("The twist angle is: ", comm_ang)
            lattice = PubMeth.tri_lattice(shell_i, a0_constant, shift_arr=array([0, 0]))
            lattice_p = [PubMeth.rotation(comm_ang / 2 + 30) @ ele_arr for ele_arr in lattice]  # plus 30 degrees to rotate the dots array.
            lattice_m = [PubMeth.rotation(-comm_ang / 2 + 30) @ ele_arr for ele_arr in lattice]
            lattice_overlap = PubMeth.overlap_points(lattice_p, lattice_m)

            K0_arr = array([a0_constant / sqrt(3), 0])
            Kp_arr = PubMeth.rotation(comm_ang / 2) @ K0_arr
            Km_arr = PubMeth.rotation(-comm_ang / 2) @ K0_arr
            Delta_K = PubMeth.rotation(comm_ang / 2) @ K0_arr - PubMeth.rotation(-comm_ang / 2) @ K0_arr
            Kp_plus_L_p = [Kp_arr + ele_arr for ele_arr in lattice_p]
            Km_plus_L_m = [Km_arr + ele_arr for ele_arr in lattice_m]
            K_overlap = PubMeth.overlap_points(Kp_plus_L_p, Km_plus_L_m)
            print("Len of K overlap is: ", len(K_overlap))
            print(cos(comm_theta / 2) * sqrt(N))
            print(sin(comm_theta / 2) * sqrt(N) * sqrt(3))

            atom_lattice_plus_set = PubMeth.plus_set(lattice_p, lattice_m, length_limit=4.5)
            tmp_plus_set = [array([round(ele_arr[0], 4), round(ele_arr[1], 4)]) for ele_arr in atom_lattice_plus_set]
            atom_lattice_plus_set = np.unique(array(tmp_plus_set), axis=0)

            if moire_figsave:
                PubMeth.draw_dots(lattice_m, marker_type='o', save_or_not=False, hold_on=True)
                PubMeth.draw_dots(lattice_p, marker_type='o', save_or_not=False, hold_on=True)
                PubMeth.draw_dots(lattice_overlap, marker_type='*', save_or_not=True, hold_on=False, fig_title=r"Atomic structure: $\theta = %.2f \degree$" % comm_ang, leg_list=[r'$P_{-}$', r'$P_{+}$', r'$P_{-} \cap P_{+}$'])
            if moire_pm_set_save:
                PubMeth.draw_dots(lattice_m, marker_type='o', save_or_not=False, hold_on=True)
                PubMeth.draw_dots(lattice_p, marker_type='o', save_or_not=False, hold_on=True)
                PubMeth.draw_dots(atom_lattice_plus_set, marker_type='.', save_or_not=True, hold_on=False, draw_width=0.1, fig_title=r"Atomic structure: $\theta = %.2f \degree$" % comm_ang, leg_list=[r'$P_{-}$', r'$P_{+}$', r'$P_{-} + P_{+}$'])
            if k_valley_save:
                PubMeth.draw_dots(Kp_plus_L_p, marker_type='o', save_or_not=False, hold_on=True, draw_width=0.1)
                PubMeth.draw_dots(Km_plus_L_m, marker_type='o', save_or_not=False, hold_on=True, draw_width=0.1)
                if len(K_overlap) != 0:
                    PubMeth.draw_dots(K_overlap, marker_type='*', save_or_not=False, hold_on=True, fig_title=r"$\theta = %.2f \degree$" % comm_ang)
                    legs = [r'$K_{-}^{0} + P_{-}$', r'$K_{+}^{0} + P_{+}$', r'$\mathcal{Q}_+$', r'$\sqrt{N}K$', r'$-\sqrt{N}K$']
                else:
                    legs = [r'$K_{-}^{0} + P_{-}$', r'$K_{+}^{0} + P_{+}$', r'$\sqrt{N}K$', r'$-\sqrt{N}K$']
                plt.plot([0, sqrt(N) * K0_arr[0]], [0, sqrt(N) * K0_arr[1]], 'r')
                plt.plot([0, -sqrt(N) * K0_arr[0]], [0, -sqrt(N) * K0_arr[1]], 'k')
                plt.title(r"K space: $\theta = %.2f \degree$" % comm_ang)
                plt.legend(legs, fontsize=7)
                plt.savefig("./tmp_figs/k_valley_{}.png".format(comm_ang), dpi=300)
                plt.close()
            if k_pm_valley_save:
                PubMeth.draw_dots(Kp_plus_L_p, marker_type='o', save_or_not=False, hold_on=True, draw_width=0.1)
                PubMeth.draw_dots(Km_plus_L_m, marker_type='o', save_or_not=False, hold_on=True, draw_width=0.1)
                if len(K_overlap) != 0:
                    PubMeth.draw_dots(K_overlap, marker_type='*', save_or_not=False, hold_on=True, fig_title=r"$\theta = %.2f \degree$" % comm_ang)
                    legs = [r'$K_{+}^{0} + P_{+}$', r'$K_{-}^{0} + P_{-}$', r'$\mathcal{Q}_+$', r'$P_{-} + P_{+}$(shifted)', r"$K_{+}^{0} - K_{+}^{'0}$", 'Origin']
                else:
                    legs = [r'$K_{+}^{0} + P_{+}$', r'$K_{-}^{0} + P_{-}$', r'$P_{-} + P_{+}$(shifted)', r"$K_{+}^{0} - K_{+}^{'0}$", 'Origin']
                PubMeth.draw_dots(atom_lattice_plus_set + array([norm(Delta_K), 0]) / sqrt(3), marker_type='.', save_or_not=False, hold_on=True, draw_width=0.1)
                plt.plot([Kp_arr[0], -Kp_arr[0]], [Kp_arr[1], -Kp_arr[1]], 'k', lw=0.4)
                plt.scatter(0, 0, marker='x', s=8)
                plt.title(r"K space: $\theta = %.2f \degree$" % comm_ang)
                plt.legend([r'$K_{+}^{0} + P_{+}$', r'$K_{-}^{0} + P_{-}$', r'$\mathcal{Q}_+$', r'$P_{-} + P_{+}$(shifted)', r"$K_{+}^{0} - K_{+}^{'0}$", 'Origin'], fontsize=6)
                plt.savefig("./tmp_figs/k_valley_and_pm_sets_{}.png".format(comm_ang), dpi=300)
                plt.close()
        else:
            if r % 3 != 0:
                N = int(3 * m0 ** 2 + 3 * m0 * r + r ** 2)
            else:
                N = int(m0 ** 2 + m0 * r + r ** 2 / 3)
            print("Number of unit cell in one supercell: ", N)

            comm_ang, comm_theta = PubMeth.commensurate_angle(m0, r)
            print("The twist angle is: ", comm_ang)
            lattice = PubMeth.tri_lattice(shell_i, a0_constant, shift_arr=array([0, 0]))
            lattice_p = [PubMeth.rotation(comm_ang / 2) @ ele_arr for ele_arr in lattice]  # plus 30 degrees to rotate the dots array.
            lattice_m = [PubMeth.rotation(-comm_ang / 2) @ ele_arr for ele_arr in lattice]
            lattice_overlap = PubMeth.overlap_points(lattice_p, lattice_m)

            K0_arr = array([a0_constant / sqrt(3), 0])
            Kp_plus_L_p = [PubMeth.rotation(comm_ang / 2) @ K0_arr + ele_arr for ele_arr in lattice_p]  # actually, K1 is the vertex of the hexagon.
            Km_plus_L_m = [PubMeth.rotation(-comm_ang / 2) @ K0_arr + ele_arr for ele_arr in lattice_m]
            K_overlap = PubMeth.overlap_points(Kp_plus_L_p, Km_plus_L_m)
            print("Len of K overlap is: ", len(K_overlap))

            atom_lattice_plus_set = PubMeth.plus_set(lattice_p, lattice_m, length_limit=4.5)
            tmp_plus_set = [array([round(ele_arr[0], 4), round(ele_arr[1], 4)]) for ele_arr in atom_lattice_plus_set]
            atom_lattice_plus_set = np.unique(array(tmp_plus_set), axis=0)

            if moire_figsave:
                PubMeth.draw_dots(lattice_m, marker_type='o', save_or_not=False, hold_on=True)
                PubMeth.draw_dots(lattice_p, marker_type='o', save_or_not=False, hold_on=True)
                PubMeth.draw_dots(lattice_overlap, marker_type='*', save_or_not=True, hold_on=False, fig_title=r"$\theta = %.2f \degree$" % comm_ang, leg_list=[r'$L_{-}$', r'$L_{+}$', r'$L_{-} \cap L_{+}$'])
            if moire_pm_set_save:
                PubMeth.draw_dots(lattice_m, marker_type='o', save_or_not=False, hold_on=True)
                PubMeth.draw_dots(lattice_p, marker_type='o', save_or_not=False, hold_on=True)
                PubMeth.draw_dots(atom_lattice_plus_set, marker_type='.', save_or_not=True, hold_on=False, draw_width=0.1, fig_title=r"$\theta = %.2f \degree$" % comm_ang, leg_list=[r'$L_{-}$', r'$L_{+}$', r'$L_{-} + L_{+}$'])
            if k_valley_save:
                PubMeth.draw_dots(Kp_plus_L_p, marker_type='o', save_or_not=False, hold_on=True, draw_width=0.1)
                PubMeth.draw_dots(Km_plus_L_m, marker_type='o', save_or_not=False, hold_on=True, draw_width=0.1)
                if len(K_overlap) != 0:
                    PubMeth.draw_dots(K_overlap, marker_type='*', save_or_not=False, hold_on=True, fig_title=r"$\theta = %.2f \degree$" % comm_ang)
                PubMeth.draw_dots(atom_lattice_plus_set, marker_type='.', save_or_not=False, hold_on=True, draw_width=0.1)
                plt.plot([0, sqrt(N) * K0_arr[0]], [0, sqrt(N) * K0_arr[1]], 'r')
                plt.plot([0, -sqrt(N) * K0_arr[0]], [0, -sqrt(N) * K0_arr[1]], 'k')
                plt.scatter(0, 0, marker='8')
                plt.title(r"K space: $\theta = %.2f \degree$" % comm_ang)
                plt.legend([r'$K_{-}^{0} + P_{-}$', r'$K_{+}^{0} + P_{+}$', r'$\mathcal{Q}_+$', r'$\sqrt{N}K$', r'$-\sqrt{N}K$'])
                plt.savefig("./tmp_figs/k_valley_{}.png".format(comm_ang), dpi=300)

    @staticmethod
    def situate_x_labels(input_path_list, density_per_path):
        dis_list = [0]
        for i in range(0, len(input_path_list) - 1):
            p1 = input_path_list[i]
            p2 = input_path_list[i + 1]
            dis_list.append(sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) * density_per_path + dis_list[-1])
        return dis_list

    @staticmethod
    def operator_d_theta(delta_angle):
        delta_theta = delta_angle / 180 * pi
        return 2 * sin(delta_theta / 2) * PubMeth.rotation(90)

    @staticmethod
    def pauli_mat(mat_index):
        sigma_0 = array([
            [1, 0],
            [0, 1]
        ])
        sigma_x = array([
            [0, 1],
            [1, 0]
        ])
        sigma_y = array([
            [0, -1j],
            [1j, 0]
        ])
        sigma_z = array([
            [1, 0],
            [0, -1]
        ])
        Pauli_mat_list = [sigma_0, sigma_x, sigma_y, sigma_z]
        return Pauli_mat_list[mat_index]
    
    @staticmethod
    def path_between_two_vec(arr_1, arr_2, density=100):
        arr_list = []
        for i_add in range(int(density)):
            tmp_arr = arr_1 + (arr_2 - arr_1) / density * i_add
            arr_list.append(tmp_arr)
        return arr_list

    @staticmethod
    def dots_around_one_point(center_vec, add_vec, density=100):
        out_arr_list = []
        add_another_dir = PubMeth.rotation(90) @ add_vec
        for i in linspace(-1, 1, density + 1):
            for j in linspace(-1, 1, density + 1):
                out_arr_list.append(center_vec + i * add_vec + j * add_another_dir)
        return out_arr_list

    @staticmethod
    def plotly_layout(xlabel='X', ylabel='Y', zlabel='Z', figuretitle='title'):
        layout = go.Layout(title=figuretitle, scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel,
            xaxis=dict(
                tickfont=dict(
                    size=14,
                    family='Old Standard TT, serif',
                )
            ),
            yaxis=dict(
                tickfont=dict(
                    size=14,
                    family='Old Standard TT, serif',
                )
            ),
            zaxis=dict(
                tickfont=dict(
                    size=14,
                    family='Old Standard TT, serif',
                )
            ),
            ))
        return layout

    @staticmethod
    def find_half_filling_energy(energies_list_in):
        dim_one_list = sorted(np.matrix.tolist(real(array(energies_list_in).reshape(1, -1)))[0])
        if len(dim_one_list) % 2 == 0:
            print("# of eigen energies is even.")
            return (dim_one_list[len(dim_one_list) // 2] + dim_one_list[len(dim_one_list) // 2 - 1]) / 2
        elif len(dim_one_list) % 2 != 0:
            print("# of eigen energies is odd.")
            return dim_one_list[len(dim_one_list) // 2]

    @staticmethod
    def plot_energies(energies_list_in, fig_size=(7, 5), y_range=[-2000, 2000], line_type='k-', figuretitle='title', x_label_pos=[], x_labs=[], save_or_not=True, show=False, save_title='band', lw=2, hold_on=False, y_label="E(meV)", test_mode=False, selected_bds_indices=[]):
        plt.figure(figsize=fig_size, dpi=300)
        if len(selected_bds_indices) == 0:
            plt.plot(real(energies_list_in), line_type, linewidth=lw)
        else:
            selected_energies_list = [real(energies_list_in)[:, bd_i] for bd_i in selected_bds_indices]
            plt.plot(array(selected_energies_list).T, line_type, linewidth=lw)
        plt.ylim(y_range)
        plt.title(figuretitle, fontsize=14)
        plt.ylabel(y_label, fontsize=12)
        plt.xticks(x_label_pos, x_labs, fontsize=12)
        if save_or_not:
            target_dir = PubMeth.get_right_save_path("bands")
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            plt.savefig(target_dir + save_title + '.png', dpi=300)
            save(target_dir + save_title + '.npy', energies_list_in)
        if not hold_on:
            if show:
                plt.show()
            plt.close()
        if test_mode:
            trace_list = []
            layout = PubMeth.plotly_layout()
            if len(selected_bds_indices) == 0:
                for i in range(array(energies_list_in).shape[1]):
                    ele_trace = go.Scatter(x=arange(array(energies_list_in).shape[0]), y=array(energies_list_in)[:, i])
                    trace_list.append(ele_trace)
                fig = go.Figure(data=trace_list, layout=layout)
                fig.write_html('eigen_energies.html')
            else:
                for ele_i in selected_bds_indices:
                    ele_trace = go.Scatter(x=arange(array(energies_list_in).shape[0]), y=array(energies_list_in)[:, ele_i])
                    trace_list.append(ele_trace)
                fig = go.Figure(data=trace_list, layout=layout)
                fig.write_html('eigen_energies.html')
        
    # # 基矢构建
    @staticmethod
    def basis_set(loop_times):  # create the basis of calculation
        def layer1to2(vec):  # (the number of b_p, the number of b_n)
            v1 = (vec[0], vec[1], 2)  # the same wave vector
            v2 = (vec[0] + 1, vec[1], 2)  # plus a b_p
            v3 = (vec[0], vec[1] + 1, 2)  # plus a b_n
            return v1, v2, v3
        def layer2to1(vec):
            v1 = (vec[0], vec[1], 1)  # the same wave vector
            v2 = (vec[0] - 1, vec[1], 1)  # minus a b_p
            v3 = (vec[0], vec[1] - 1, 1)  # minus a b_n
            return v1, v2, v3
        original_basis_list = [(0, 0, 1)]
        times = 0
        vec_list_c = [(0, 0, 1)]
        while times < loop_times:
            for ele_vec in original_basis_list:
                if ele_vec[2] == 1:
                    v1, v2, v3 = layer1to2(ele_vec)
                    vec_list_c.extend([v1, v2, v3])
            times = times + 1
            original_basis_list = vec_list_c[:]
            if times < loop_times:
                for ele_vec in original_basis_list:
                    if ele_vec[2] == 2:
                        v1, v2, v3 = layer2to1(ele_vec)
                        vec_list_c.extend([v1, v2, v3])
                times = times + 1
            original_basis_list = vec_list_c[:]
        output_list = []
        for element in original_basis_list:
            if element in output_list:
                pass
            else:
                output_list.append(element)
        return output_list

    @staticmethod
    def sigma_angle_dot_p(p_arr, angle_in):
        theta_in = angle_in / 180 * pi
        p_x = p_arr[0]
        p_y = p_arr[1]
        return array([
            [0, exp(-1j * theta_in) * (p_x - 1j * p_y)],
            [exp(1j * theta_in) * (p_x + 1j * p_y), 0]
        ])

    @staticmethod
    def save_if_same_name(func_save, save_name, target_dir='./', splitter='_'):
        full_save_name = target_dir + save_name
        if os.path.exists(target_dir+save_name):
            flag_i = 1
            splitted_elements = full_save_name.split(splitter)
            splitted_elements[-1] = '#1' + splitted_elements[-1]
            while True:
                if os.path.exists('_'.join(splitted_elements)):
                    flag_i = flag_i + 1
                    splitted_elements[-1] = '#{:.0f}'.format(flag_i) + splitted_elements[-1][2:]
                else:
                    break
            func_save('_'.join(splitted_elements))
        else:
            func_save(target_dir + save_name)
        