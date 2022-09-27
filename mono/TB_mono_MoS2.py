import time

import matplotlib.pyplot as plt
from numpy import array, sqrt, conj, kron, block, zeros, cos, sin, eye, pi, linspace, dot, exp, arange, argsort
from numpy.linalg import eig, norm

from mono.TB_for_mono_tmds import MonoMoS2, PubMeth
from scipy.special import struve, yn
from public.consts import *


class MoS2(MonoMoS2):
    # # all parameters are in unit of eV
    O1 = 3.558
    t = -0.189
    t_pri = -0.117
    txy = 0.024
    s = -0.041
    s_pri = 0.003
    u = 0.165
    u_pri = -0.122
    uxy = -0.140

    def __init__(self, lam=0.073, diele_const=2.5):
        super().__init__(par_type="GGA")
        self.lam = lam
        self.diele_const = diele_const  # CGS unit
        self.r0_ratio = 33.875 / self.diele_const
        self.R1 = self.a * array([1, 0])
        self.R2 = self.a * array([1 / 2, sqrt(3) / 2])
        self.b1 = self.Kg * sqrt(3) * array([sqrt(3) / 2, -1 / 2])
        self.b2 = self.Kg * sqrt(3) * array([0, 1])

    def soc_mat(self):  # eV for lam
        L = array([
            [0, 0, 0, 0, 2j * sqrt(3)],
            [0, 0, 2j, -2j, 0],
            [0, -2j, 0, 0, 2j],
            [0, 2j, 0, 0, -1j],
            [-2j * sqrt(3), 0, -2j, 1j, 0]
        ])
        L2 = array([
            [0, 0, 0, 0, 0],
            [0, 0, 2j, 0, 0],
            [0, -2j, 0, 0, 0],
            [0, 0, 0, 0, -1j],
            [0, 0, 0, 1j, 0]
        ])
        S = array([
            [1, 0],
            [0, -1]
        ])
        return self.lam / 2 * kron(S, L2)

    def H0_even(self, k_arr):
        return super().hamiltonian(k_arr)

    def H0_odd(self, k_arr):
        alpha = 1 / 2 * k_arr[0] * self.a
        beta = sqrt(3) / 2 * k_arr[1] * self.a

        def hx():
            return self.O1 + 2 * self.t * cos(2 * alpha) + (self.t + 3 * self.t_pri) * cos(alpha) * cos(beta) \
                + 4 * self.s * cos(3 * alpha) * cos(beta) + (3 * self.s_pri - self.s) * cos(2 * beta) \
                + 2 * self.u * cos(4 * alpha) + (self.u + 3 * self.u_pri) * cos(2 * alpha) * cos(2 * beta)

        def hy():
            return self.O1 + 2 * self.t_pri * cos(2 * alpha) + (self.t_pri + 3 * self.t) * cos(alpha) * cos(beta) \
                + 4 * self.s_pri * cos(3 * alpha) * cos(beta) + (3 * self.s - self.s_pri) * cos(2 * beta) \
                + 2 * self.u_pri * cos(4 * alpha) + (self.u_pri + 3 * self.u) * cos(2 * alpha) * cos(2 * beta)

        def hxy():
            return 4j * self.txy * sin(alpha) * (cos(alpha) - cos(beta)) \
                + sqrt(3) * (self.t_pri - self.t) * sin(alpha) * sin(beta) \
                + 2 * sqrt(3) * (self.s_pri - self.s) * sin(alpha) * sin(beta) * (1 + 2 * cos(2 * alpha)) \
                + 4j * self.uxy * sin(2 * alpha) * (cos(2 * alpha) - cos(2 * beta)) \
                + sqrt(3) * (self.u_pri - self.u) * sin(2 * alpha) * sin(2 * beta)

        return array([
            [hx(), hxy()],
            [conj(hxy()), hy()]
        ])

    def H0(self, k_arr):
        return block([
            [self.H0_even(k_arr), zeros((3, 2))],
            [zeros((2, 3)), self.H0_odd(k_arr)]
        ])

    def hamiltonian(self, k_arr):
        return self.soc_mat() + kron(eye(2), self.H0(k_arr))

    def k_R_list(self, p_density):
        list_kp = []
        list_R = []
        for x1 in linspace(0, 1, p_density):
            for x2 in linspace(0, 1, p_density):
                list_kp.append(x1 * self.b1 + x2 * self.b2)
        for x1 in arange(p_density):
            for x2 in arange(p_density):
                list_R.append(x1 * self.R1 + x2 * self.R2)
        return list_kp, list_R

    def V_R(self, R_in):
        func_const = pi * e_cgs ** 2 * erg2eV * cm2m * m2A / (2 * self.diele_const * self.r0_ratio)
        return func_const * (struve(0, R_in / self.r0_ratio) - yn(0, R_in / self.r0_ratio))

    def V_q(self, R_arr_list, q_arr_in):
        dis_list = [norm(ele_arr) for ele_arr in R_arr_list]
        dis_list.sort()
        V_R = [self.V_R(ele_dist) for ele_dist in dis_list]
        V_R[0] = self.V_R(self.a)
        exp_list = [exp(1j * dot(q_arr_in, ele_arr)) for ele_arr in R_arr_list]
        out_result = array(exp_list) * array(V_R)
        return out_result.sum()

    def mel_k1k2(self, Q_arr, k1, k2, vc_basis, R_arr_list):
        h1 = self.hamiltonian(k1)
        h2 = self.hamiltonian(k2)
        e_v1, e_a1 = eig(h1)
        e_a1 = e_a1.T[argsort(e_v1)]
        e_v1.sort()
        e_v2, e_a2 = eig(h2)
        e_a2 = e_a2.T[argsort(e_v1)]
        e_v1.sort()

        h1Q = self.hamiltonian(k1 + Q_arr)
        h2Q = self.hamiltonian(k2 + Q_arr)
        e_v1Q, e_a1Q = eig(h1Q)
        e_a1Q = e_a1Q.T[argsort(e_v1Q)]
        e_v1Q.sort()
        e_v2Q, e_a2Q = eig(h2Q)
        e_a2Q = e_a2Q.T[argsort(e_v1Q)]
        e_v1Q.sort()

        V_kk = self.V_q(R_arr_list, k1 - k2)
        V_Q = self.V_q(R_arr_list, Q_arr)

        out_mat = []
        for bra_v in vc_basis:
            row = []
            for ket_v in vc_basis:
                # # here, bra_v=ket_v=(v_index, c_index)
                D_m1 = conj(e_a1Q[bra_v[1]]) @ e_a2Q[ket_v[1]]
                D_m2 = conj(e_a2[ket_v[0]]) @ e_a1[bra_v[0]]
                X_m1 = conj(e_a1Q[bra_v[1]]) @ e_a1[bra_v[0]]
                X_m2 = conj(e_a2[ket_v[0]]) @ e_a2Q[ket_v[1]]
                D = 1 / len(R_arr_list) * V_kk * D_m1 * D_m2
                X = 1 / len(R_arr_list) * V_Q * X_m1 * X_m2
                if bra_v[0] == ket_v[0] and bra_v[1] == ket_v[1] and k1[0] == k2[0] and k1[1] == k2[1]:
                    ele_e = e_v1Q[bra_v[1]] - e_v1[bra_v[0]] - (D - X)
                    row.append(ele_e)
                else:
                    row.append(-(D - X))
            out_mat.append(row)
        return array(out_mat)

    def exciton_h(self, p_density, Q_arr):
        v_indices = arange(2)
        c_indices = arange(2, 10)
        vc_basis = []
        for ele_v in v_indices:
            for ele_c in c_indices:
                vc_basis.append((ele_v, ele_c))
        k_list, R_list = self.k_R_list(p_density)
        out_mat = []
        for k1 in k_list:
            row = []
            for k2 in k_list:
                row.append(self.mel_k1k2(Q_arr, k1, k2, vc_basis, R_list))
            out_mat.append(row)

        return block(out_mat)


def main():
    mos2 = MoS2()
    t1 = time.time()
    # tmp_h = mos2.exciton_h(6, array([0., 0.]))
    # tmp_e, tmp_a = eig(tmp_h)
    # tmp_e.sort()
    # print(tmp_e)
    # print("One sole construction: ", time.time() - t1)
    # # # kpoints construction
    # k_path = [mos2.path_gamma, mos2.path_K1, mos2.path_M2, mos2.path_gamma]
    # k_path = [-mos2.path_M, -mos2.path_K1, mos2.path_gamma, mos2.path_K1, mos2.path_M]
    # k_path = [mos2.path_K1, mos2.path_gamma, mos2.path_M, mos2.path_K2]
    # kp_list = []
    # for i in range(len(k_path) - 1):
    #     kp_list.extend(PubMeth.p2p(k_path[i], k_path[i + 1], 100))
    #
    # # path depiction
    # k_path = [mos2.path_K1, mos2.path_gamma, mos2.path_M, mos2.path_K2]
    # kp_list = []
    # for i in range(len(k_path) - 1):
    #     kp_list.extend(PubMeth.p2p(k_path[i], k_path[i + 1], 100))
    # eig_list = []
    # for ele_kp in kp_list:
    #     tmp_e, tmp_a = eig(mos2.hamiltonian(ele_kp))
    #     tmp_e.sort()
    #     eig_list.append(tmp_e)
    # print(len(eig_list[0]))
    # plt.ylabel("E(eV)")
    # plt.plot(eig_list)
    # plt.show()

    # # berry curvature calculation
    # berry_result = []
    # for ele_kp in kp_list:
    #     berry_result.append(mos2.berry_cur(ele_kp, 0))
    # print(berry_result)
    # plt.plot(berry_result)
    # plt.xticks([])
    # plt.ylim([-20, 20])
    # plt.ylabel("Berry Curvature")
    # plt.show()

    # Vq_result = []
    # mos2.V_q(10, array([0, 0]))
    k_list, R_list = mos2.k_R_list(20)
    plt.scatter(array(R_list)[:, 0], array(R_list)[:, 1])
    # plt.scatter(array(k_list)[:, 0], array(k_list)[:, 1])
    print("Lattice num: ", len(R_list), len(k_list))
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.show()

    # n_cell_list = []
    # for shell_i in range(1, 100):
    #     sv1 = shell_i * array([-1 / 2, sqrt(3) / 2]) - shell_i * array([1, 0])
    #     sv2 = shell_i * array([-1 / 2, -sqrt(3) / 2]) - shell_i * array([1, 0])
    #     s_area = sqrt(3) / 2 * norm(sv1) ** 2
    #     unit_area = sqrt(3) / 2
    #     n_cell = s_area / unit_area
    #     n_cell_list.append(n_cell)
    # print(n_cell_list)
    # print(45 ** 2)
    # plt.plot(n_cell_list)
    # plt.xlabel("Shell num", fontsize=14)
    # plt.ylabel("Number of cells", fontsize=14)
    # plt.show()

    shell_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for mesh_i in shell_list:
        tmp = mos2.V_q(mesh_i, array([0., 0.]))
        sv1 = mesh_i * array([-1 / 2, sqrt(3) / 2]) - mesh_i * array([1, 0])
        s_area = sqrt(3) / 2 * norm(sv1) ** 2
        unit_area = sqrt(3) / 2
        n_cell = s_area / unit_area
        Vq_result.append(tmp / n_cell)
        print("Complete: ", mesh_i)
    plt.plot(shell_list, Vq_result)
    plt.xlabel("Shell", fontsize=12)
    plt.ylabel(r"$\frac{1}{N} \sum e^{i \vec{q} \cdot \vec{R}} V_R$", fontsize=12)
    plt.title(r"$\vec{q}=0$", fontsize=14)
    plt.show()

    result = []
    r_list = linspace(1, 300, 100)
    for r in r_list:
        result.append(mos2.V_R(r))
    plt.plot(r_list, result)
    plt.show()


if __name__ == "__main__":
    main()
