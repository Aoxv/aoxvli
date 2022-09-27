import matplotlib.pyplot as plt
from numpy import sqrt, cos, sin, array, conj, pi, argsort, real
from numpy.linalg import eig

from public.consts import m2A
from public.method import PubMeth
e_cgs = 4.80320427e-10  # statC = 1 erg^1/2 * cm^1/2
erg2J = 1e-7  # J
erg2eV = 6.2415e11  # eV
cm2m = 1e-2


class ThirdNN:

    def __init__(self, a, epsilon1, epsilon2, t0, t1, t2, t11, t12, t22, r0, r2, r11, r12, u0, u1, u2, u11, u12, u22, r1):
        self.a = a
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
        self.t11 = t11
        self.t12 = t12
        self.t22 = t22
        self.r0 = r0
        self.r2 = r2
        self.r11 = r11
        self.r12 = r12
        self.u0 = u0
        self.u1 = u1
        self.u2 = u2
        self.u11 = u11
        self.u12 = u12
        self.u22 = u22
        self.r1 = r1

        self.Kg = 4 * pi / (3 * a)
        self.path_gamma = array([0, 0])
        self.path_K1 = self.Kg * array([1, 0])
        self.path_K2 = self.Kg * array([1 / 2, sqrt(3) / 2])
        self.path_K2_equal = self.Kg * array([1 / 2, -sqrt(3) / 2])
        self.path_M = (self.path_K1 + self.path_K2) / 2
        self.path_M2 = (self.path_K1 + self.path_K2_equal) / 2

    def hamiltonian(self, k_arr):
        alpha = 1 / 2 * k_arr[0] * self.a
        beta = sqrt(3) / 2 * k_arr[1] * self.a

        def V0():
            return self.epsilon1 + 2 * self.t0 * (2 * cos(alpha) * cos(beta) + cos(2 * alpha)) \
                + 2 * self.r0 * (2 * cos(3 * alpha) * cos(beta) + cos(2 * beta)) \
                + 2 * self.u0 * (2 * cos(2 * alpha) * cos(2 * beta) + cos(4 * alpha))

        def ReV1():
            return -2 * sqrt(3) * self.t2 * sin(alpha) * sin(beta) + 2 * (self.r1 + self.r2) * sin(3 * alpha) * sin(beta) \
                - 2 * sqrt(3) * self.u2 * sin(2 * alpha) * sin(2 * beta)

        def ImV1():
            return 2 * self.t1 * sin(alpha) * (2 * cos(alpha) + cos(beta)) + 2 * (self.r1 - self.r2) * sin(3 * alpha) * cos(beta) \
                + 2 * self.u1 * sin(2 * alpha) * (2 * cos(alpha) + cos(2 * beta))

        def ReV2():
            return 2 * self.t2 * (cos(2 * alpha) - cos(alpha) * cos(beta)) \
                - 2 / sqrt(3) * (self.r1 + self.r2) * (cos(3 * alpha) * cos(beta) - cos(2 * beta)) \
                + 2 * self.u2 * (cos(4 * alpha) - cos(2 * alpha) * cos(2 * beta))

        def ImV2():
            return 2 * sqrt(3) * self.t1 * cos(alpha) * sin(beta) \
                + 2 / sqrt(3) * sin(beta) * (self.r1 - self.r2) * (cos(3 * alpha) + 2 * cos(beta)) \
                + 2 * sqrt(3) * self.u1 * cos(2 * alpha) * sin(2 * beta)

        def V11():
            return self.epsilon2 + (self.t11 + 3 * self.t22) * cos(alpha) * cos(beta) + 2 * self.t11 * cos(2 * alpha) \
                + 4 * self.r11 * cos(3 * alpha) * cos(beta) + 2 * (self.r11 + sqrt(3) * self.r12) * cos(2 * beta) \
                + (self.u11 + 3 * self.u22) * cos(2 * alpha) * cos(2 * beta) + 2 * self.u11 * cos(4 * alpha)

        def ReV12():
            return sqrt(3) * (self.t22 - self.t11) * sin(alpha) * sin(beta) + 4 * self.r12 * sin(3 * alpha) * sin(beta) \
                + sqrt(3) * (self.u22 - self.u11) * sin(2 * alpha) * sin(2 * beta)

        def ImV12():
            return 4 * self.t12 * sin(alpha) * (cos(alpha) - cos(beta)) \
                + 4 * self.u12 * sin(2 * alpha) * (cos(2 * alpha) - cos(2 * beta))

        def V22():
            return self.epsilon2 + (3 * self.t11 + self.t22) * cos(alpha) * cos(beta) + 2 * self.t22 * cos(2 * alpha) \
                + 2 * self.r11 * (2 * cos(3 * alpha) * cos(beta) + cos(2 * beta)) \
                + 2 / sqrt(3) * self.r12 * (4 * cos(3 * alpha) * cos(beta) - cos(2 * beta)) \
                + (3 * self.u11 + self.u22) * cos(2 * alpha) * cos(2 * beta) + 2 * self.u22 * cos(4 * alpha)

        V1 = ReV1() + ImV1() * 1j
        V2 = ReV2() + ImV2() * 1j
        V12 = ReV12() + ImV12() * 1j

        return array([
            [V0(), V1, V2],
            [conj(V1), V11(), V12],
            [conj(V2), conj(V12), V22()]
        ])

    def berry_cur(self, k_arr, band_i):
        interval_k = 0.00001

        cent_h = self.hamiltonian(k_arr)
        e_v, e_a = eig(cent_h)
        vector_n = e_a.T[argsort(real(e_v))[band_i]]
        e_n = e_v[argsort(real(e_v))[band_i]]

        other_n_pri = list(e_a.T)
        other_e = list(e_v)
        other_n_pri.pop(argsort(real(e_v))[band_i])
        other_e.pop(argsort(real(e_v))[band_i])

        par_h_kx = (self.hamiltonian(array([k_arr[0] + interval_k, k_arr[1]])) - cent_h) / interval_k
        par_h_ky = (self.hamiltonian(array([k_arr[0], k_arr[1] + interval_k])) - cent_h) / interval_k

        out_result = 0
        for ele_i in range(len(other_e)):
            tmp_e = other_e[ele_i]
            tmp_n_pri = other_n_pri[ele_i]

            mat_n_n_pri = tmp_n_pri.reshape(-1, 1) @ conj(tmp_n_pri.reshape(1, -1))
            cent_mat_xy = par_h_kx @ mat_n_n_pri @ par_h_ky
            cent_mat_yx = par_h_ky @ mat_n_n_pri @ par_h_kx
            term_xy = conj(vector_n.reshape(1, -1)) @ cent_mat_xy @ vector_n
            term_yx = conj(vector_n.reshape(1, -1)) @ cent_mat_yx @ vector_n
            out_result = out_result + (term_xy - term_yx) / (real(e_n - tmp_e)) ** 2

        return out_result * 1j


class MonoMoS2(ThirdNN):
    params_GGA = [3.190, 0.683, 1.707, -0.146, -0.114, 0.506, 0.085, 0.162, 0.073, 0.060,
                  0.067, 0.016, 0.087, -0.038, 0.046, 0.001, 0.266, -0.176, -0.150, -0.236]
    params_LDA = [3.129, 0.820, 1.931, -0.176, -0.101, 0.531, 0.084, 0.169, 0.070, 0.070,
                  0.084, 0.019, 0.093, -0.043, 0.047, 0.005, 0.304, -0.192, -0.162, -0.252]

    def __init__(self, par_type="GGA"):
        if par_type == "GGA":
            super().__init__(*self.params_GGA)
        elif par_type == "LDA":
            super().__init__(*self.params_LDA)


class MonoWS2(ThirdNN):
    params_GGA = [3.191, 0.717, 1.916, -0.152, -0.097, 0.590, 0.047, 0.178, 0.016, 0.069,
                  0.107, -0.003, 0.109, -0.054, 0.045, 0.002, 0.325, -0.206, -0.163, -0.261]
    params_LDA = [3.132, 0.905, 2.167, -0.175, -0.090, 0.611, 0.043, 0.181, 0.008, 0.075,
                  0.127, 0.001, 0.114, -0.063, 0.047, 0.004, 0.374, -0.224, -0.177, -0.282]

    def __init__(self, par_type="GGA"):
        if par_type == "GGA":
            super().__init__(*self.params_GGA)
        elif par_type == "LDA":
            super().__init__(*self.params_LDA)


class MonoMoSe2(ThirdNN):
    params_GGA = [3.326, 0.684, 1.546, -0.146, -0.130, 0.432, 0.144, 0.117, 0.075, 0.039,
                  0.069, 0.052, 0.060, -0.042, 0.036, 0.008, 0.272, -0.172, -0.150, -0.209]
    params_LDA = [3.254, 0.715, 1.687, -0.154, -0.134, 0.437, 0.124, 0.119, 0.072, 0.048,
                  0.090, 0.066, 0.045, -0.067, 0.041, 0.005, 0.327, -0.194, -0.151, -0.248]

    def __init__(self, par_type="GGA"):
        if par_type == "GGA":
            super().__init__(*self.params_GGA)
        elif par_type == "LDA":
            super().__init__(*self.params_LDA)


class MonoWSe2(ThirdNN):
    params_GGA = [3.325, 0.728, 1.655, -0.146, -0.124, 0.507, 0.117, 0.127, 0.015, 0.036,
                  0.107, 0.044, 0.075, -0.061, 0.032, 0.007, 0.329, -0.202, -0.164, -0.234]
    params_LDA = [3.253, 0.860, 1.892, -0.152, -0.125, 0.508, 0.094, 0.129, 0.009, 0.044,
                  0.129, 0.059, 0.058, -0.090, 0.039, 0.001, 0.392, -0.224, -0.165, -0.278]

    def __init__(self, par_type="GGA"):
        if par_type == "GGA":
            super().__init__(*self.params_GGA)
        elif par_type == "LDA":
            super().__init__(*self.params_LDA)


class MonoMoTe2(ThirdNN):
    params_GGA = [3.557, 0.588, 1.303, -0.226, -0.234, 0.036, 0.400, 0.098, 0.017, 0.003,
                  -0.169, 0.082, 0.051, 0.057, 0.103, 0.187, -0.045, -0.141, 0.087, -0.025]
    params_LDA = [3.472, 0.574, 1.410, -0.148, -0.173, 0.333, 0.203, 0.186, 0.127, 0.007,
                  0.067, 0.073, 0.081, -0.054, 0.008, 0.037, 0.145, -0.078, 0.035, -0.280]

    def __init__(self, par_type="GGA"):
        if par_type == "GGA":
            super().__init__(*self.params_GGA)
        elif par_type == "LDA":
            super().__init__(*self.params_LDA)


class MonoWTe2(ThirdNN):
    params_GGA = [3.560, 0.697, 1.380, -0.109, -0.164, 0.368, 0.204, 0.093, 0.038, -0.015,
                  0.107, 0.115, 0.009, -0.066, 0.011, -0.013, 0.312, -0.177, -0.132, -0.209]
    params_LDA = [3.476, 0.675, 1.489, -0.124, -0.159, 0.362, 0.196, 0.101, 0.044, -0.009,
                  0.129, 0.131, -0.007, -0.086, 0.012, -0.020, 0.361, -0.193, -0.129, -0.250]

    def __init__(self, par_type="GGA"):
        if par_type == "GGA":
            super().__init__(*self.params_GGA)
        elif par_type == "LDA":
            super().__init__(*self.params_LDA)


def main():
    mos2 = MonoMoS2("GGA")
    print()

    # # kpoints construction
    k_path = [mos2.path_gamma, mos2.path_K1, mos2.path_M2, mos2.path_gamma]
    k_path = [-mos2.path_M, -mos2.path_K1, mos2.path_gamma, mos2.path_K1, mos2.path_M]
    kp_list = []
    for i in range(len(k_path) - 1):
        kp_list.extend(PubMeth.p2p(k_path[i], k_path[i + 1], 100))

    # # path depiction
    # k_path = [mos2.path_gamma, mos2.path_K1, mos2.path_M2, mos2.path_gamma]
    # kp_list = []
    # for i in range(len(k_path) - 1):
    #     kp_list.extend(PubMeth.p2p(k_path[i], k_path[i + 1], 100))
    # eig_list = []
    # for ele_kp in kp_list:
    #     tmp_e, tmp_a = eig(mos2.hamiltonian(ele_kp))
    #     tmp_e.sort()
    #     eig_list.append(tmp_e)
    # print(len(eig_list))
    # plt.plot(eig_list)
    # plt.show()

    # # # berry curvature
    # berry_result = []
    # for ele_kp in kp_list:
    #     berry_result.append(mos2.berry_cur(ele_kp, 0))
    # print(berry_result)
    # plt.plot(berry_result)
    # plt.xticks([])
    # plt.ylim([-20, 20])
    # plt.ylabel("Berry Curvature")
    # plt.show()




if __name__ == "__main__":
    main()
