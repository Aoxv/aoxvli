import matplotlib.pyplot as plt
from numpy import eye

from twisted_mat.twisted_bi_gra import *

a0 = 3.472  # A, TMD lattice constant


class ContiTmdRemoCond(ContiTbgInst):

    def __init__(self, twist_angle, Delta_g=1100, V_lu=5.97, V_rd=8.0, psi_lu=-87.9 / 180 * pi,
                 psi_rd=-89.6 / 180 * pi, w_lu=-2.0, w_rd=-8.5, w_ru=15.3+0j, w_ld=15.3+0j, basis_loop_times=8):
        super().__init__(twist_angle, a0_constant=3.472, v_F=0.4e6)
        self.Delta_g = Delta_g
        self.V_lu = V_lu
        self.V_rd = V_rd
        self.psi_lu = psi_lu
        self.psi_rd = psi_rd
        self.w = array([
            [w_lu, w_ru],
            [w_ld, w_rd]
        ])
        self.pre_basis_list = self.basis_set(basis_loop_times)
        
        self.mat_type = 't_MoTe2'

    def T_p(self):
        return array([
            [self.V_lu * exp(1j * self.psi_lu), 0],
            [0, self.V_rd * exp(1j * self.psi_rd)]
        ])

    def T_m(self):
        return array([
            [self.V_lu * exp(-1j * self.psi_lu), 0],
            [0, self.V_rd * exp(-1j * self.psi_rd)]
        ])

    def mass_mat(self):
        return array([
            [self.Delta_g, 0],
            [0, 0]
        ])

    @staticmethod
    def neighbor_k_b(vec):
        g1 = (vec[0] + 1, vec[1] - 1, vec[2])
        g3 = (vec[0], vec[1] + 1, vec[2])
        g5 = (vec[0] - 1, vec[1], vec[2])

        g2 = (vec[0] + 1, vec[1], vec[2])
        g4 = (vec[0] - 1, vec[1] + 1, vec[2])
        g6 = (vec[0], vec[1] - 1, vec[2])
        return [g1, g3, g5], [g2, g4, g6]  # first list for T_m, second list for T_p

    def hamiltonian_construction(self, k):
        h = []
        for bra_v in self.pre_basis_list:
            if bra_v[2] == 1:
                h_r = []
                v1, v2, v3 = self.layer1to2(bra_v)
                t_m_list, t_p_list = self.neighbor_k_b(bra_v)
                for ket_v in self.pre_basis_list:
                    if ket_v == bra_v:
                        h_r.append(self.h_b(k + ket_v[0] * self.b_p_arr + ket_v[1] * self.b_n_arr) + self.mass_mat())
                    elif ket_v == v1:
                        h_r.append(self.t_0())
                    elif ket_v == v2:
                        h_r.append(self.t_p1())
                    elif ket_v == v3:
                        h_r.append(self.t_n1())
                    elif ket_v in t_m_list:
                        h_r.append(self.T_m())
                    elif ket_v in t_p_list:
                        h_r.append(self.T_p())
                    else:
                        h_r.append(np.zeros((2, 2)))
                h.append(h_r)
            if bra_v[2] == 2:
                h_r = []
                v1, v2, v3 = self.layer2to1(bra_v)
                t_m_list, t_p_list = self.neighbor_k_b(bra_v)
                for ket_v in self.pre_basis_list:
                    if ket_v == bra_v:
                        h_r.append(self.h_t(k + ket_v[0] * self.b_p_arr + ket_v[1] * self.b_n_arr) + self.mass_mat())
                    elif ket_v == v1:
                        h_r.append(conj(self.t_0().T))
                    elif ket_v == v2:
                        h_r.append(conj(self.t_p1().T))
                    elif ket_v == v3:
                        h_r.append(conj(self.t_n1().T))
                    elif ket_v in t_p_list:
                        h_r.append(self.T_m())
                    elif ket_v in t_m_list:
                        h_r.append(self.T_p())
                    else:
                        h_r.append(np.zeros((2, 2)))
                h.append(h_r)
        return np.block(h)


class ContiTmdRemoVal(ContiTmdRemoCond):

    def __init__(self, twist_angle, Delta_SOC=220.5):
        super().__init__(twist_angle, V_rd=7.7, V_lu=8, psi_rd=-88.35 / 180 * pi, psi_lu=-89.6 / 180 * pi,
                         w_rd=-6, w_lu=-8.5, w_ru=-1j * 5.6, w_ld=-1j * 5.6)
        self.m_star = self.Delta_g / (2 * self.v_F ** 2)
        self.Delta_SOC = Delta_SOC

    def h_b(self, k):
        x_wave = k[0] + sqrt(3) / 2
        y_wave = k[1] + 1 / 2
        modulo = abs((x_wave + 1j * y_wave) * self.norm_Kg_conti)
        core_e = - h_bar_eV ** 2 * modulo ** 2 / (2 * self.m_star) * 1e26
        return eye(2) * core_e

    def h_t(self, k):
        x_wave = k[0] + sqrt(3) / 2
        y_wave = k[1] - 1 / 2
        modulo = abs((x_wave + 1j * y_wave) * self.norm_Kg_conti)
        core_e = - h_bar_eV ** 2 * modulo ** 2 / (2 * self.m_star) * 1e26
        return eye(2) * core_e

    def soc_coup(self):
        return array([
            [0, 0],
            [0, -self.Delta_SOC]
        ])

    def hamiltonian_construction(self, k):  # construct hamiltonian matrix
        h = []
        for bra_v in self.pre_basis_list:
            if bra_v[2] == 1:
                h_r = []
                v1, v2, v3 = self.layer1to2(bra_v)
                t_m_list, t_p_list = self.neighbor_k_b(bra_v)
                for ket_v in self.pre_basis_list:
                    if ket_v == bra_v:
                        h_r.append(self.h_b(k + ket_v[0] * self.b_p_arr + ket_v[1] * self.b_n_arr) + self.soc_coup())
                    elif ket_v == v1:
                        h_r.append(self.t_0())
                    elif ket_v == v2:
                        h_r.append(self.t_p1())
                    elif ket_v == v3:
                        h_r.append(self.t_n1())
                    elif ket_v in t_m_list:
                        h_r.append(self.T_m())
                    elif ket_v in t_p_list:
                        h_r.append(self.T_p())
                    else:
                        h_r.append(np.zeros((2, 2)))
                h.append(h_r)
            if bra_v[2] == 2:
                h_r = []
                v1, v2, v3 = self.layer2to1(bra_v)
                t_m_list, t_p_list = self.neighbor_k_b(bra_v)
                for ket_v in self.pre_basis_list:
                    if ket_v == bra_v:
                        h_r.append(self.h_t(k + ket_v[0] * self.b_p_arr + ket_v[1] * self.b_n_arr) + self.soc_coup())
                    elif ket_v == v1:
                        h_r.append(conj(self.t_0().T))
                    elif ket_v == v2:
                        h_r.append(conj(self.t_p1().T))
                    elif ket_v == v3:
                        h_r.append(conj(self.t_n1().T))
                    elif ket_v in t_p_list:
                        h_r.append(self.T_m())
                    elif ket_v in t_m_list:
                        h_r.append(self.T_p())
                    else:
                        h_r.append(np.zeros((2, 2)))
                h.append(h_r)
        return np.block(h)


def main():
    a = ContiTmdRemoCond(1.2)
    vec_list = a.basis_set(7)
    print(len(vec_list))
    # a.plot_along_path(0, 0, [-1000, 1000], test_mode=True, shift_or_not=False, selected_bds=arange(len(a.pre_basis_list) - 10, len(a.pre_basis_list)))
    # path = [a.k_b_arr, a.gamma0_arr, a.k_t_arr, a.k_b_arr, (sqrt(3) / 2, -1 / 2)]
    # labs = [r"$K_+$", r"$\Gamma$", r"$K_-$", r"$K_+$", r"$K_+'$"]
    # input_args = [len(vec_list)-2, 20]
    # print(a.multi_proc_chern_num_jmeth(input_args))
    # eig_list = a.multi_proc_path(path, vec_list)
    # plt.plot(array(eig_list)[:, -5:])
    # plt.show()


if __name__ == "__main__":
    main()
