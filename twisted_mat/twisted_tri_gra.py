from twisted_mat.twisted_bi_gra import *


class ContiTtgInst(ContiTbgInst):

    moire_reci_vecs = [(1, 0), (-1, 0), (0, -1), (1, -1), (-1, 1), (0, 1)]

    def __init__(
            self,
            twist_angle,
            twist_l_index,
            res_l_index,
            base_intra_coup,
            res_intra_coup,
            twist_intra_coup,
            inter_coup_mat_near1,
            inter_coup_mat_near2,
            inter_coup_mat_near3,
            inter_coup_mat_far1,
            inter_coup_mat_far2,
            inter_coup_mat_far3,
            orient_coup,
            orient_reci_coup_g1,
            intra_reci_coup_g1_base,
            intra_reci_coup_g1_res,
            intra_reci_coup_g1_twist,
            stack_label,
            raman_gamma=100,
            kp_num=70,
            base_l_index=1,
            basis_loop_times=3):
        self.twist_l_index = twist_l_index
        self.res_l_index = res_l_index
        self.base_intra_coup = base_intra_coup
        self.res_intra_coup = res_intra_coup
        self.twist_intra_coup = twist_intra_coup
        self.inter_coup_mat_near1 = inter_coup_mat_near1
        self.inter_coup_mat_near2 = inter_coup_mat_near2
        self.inter_coup_mat_near3 = inter_coup_mat_near3
        self.inter_coup_mat_far1 = inter_coup_mat_far1
        self.inter_coup_mat_far2 = inter_coup_mat_far2
        self.inter_coup_mat_far3 = inter_coup_mat_far3
        self.orient_coup = orient_coup
        self.orient_reci_coup_g1 = orient_reci_coup_g1
        self.intra_reci_coup_g1_base = intra_reci_coup_g1_base
        self.intra_reci_coup_g1_res = intra_reci_coup_g1_res
        self.intra_reci_coup_g1_twist = intra_reci_coup_g1_twist
        self.stack_label = stack_label
        self.base_l_index = base_l_index

        super().__init__(twist_angle, raman_gamma=raman_gamma, kp_num=kp_num)
        self.pre_basis_list = self.basis_set(basis_loop_times)

    def which_reci(self, bra_v, ket_v):
        diff_vec = (ket_v[0] - bra_v[0], ket_v[1] - bra_v[1])
        reci_index = self.moire_reci_vecs.index(diff_vec)
        return reci_index

    @staticmethod
    def intra_reci_coup_mat_list(intra_reci_coup_g1):
        # for g2
        intra_reci_coup_g2 = conj(intra_reci_coup_g1.copy()).T

        # for g3
        intra_reci_coup_g3 = intra_reci_coup_g1.copy() * \
            array([
                [1, exp(1j * 2 * pi / 3)],
                [exp(-1j * 2 * pi / 3), 1]
            ])

        # for g4
        intra_reci_coup_g4 = intra_reci_coup_g1.copy() * \
            array([
                [1, exp(-1j * 2 * pi / 3)],
                [exp(1j * 2 * pi / 3), 1]
            ])
        np.fill_diagonal(
            intra_reci_coup_g4, conj(
                np.diag(
                    intra_reci_coup_g1)))

        # for g5
        intra_reci_coup_g5 = conj(intra_reci_coup_g4.copy()).T

        # for g6
        intra_reci_coup_g6 = conj(intra_reci_coup_g3.copy()).T

        return intra_reci_coup_g1, intra_reci_coup_g2, intra_reci_coup_g3, intra_reci_coup_g4, intra_reci_coup_g5, intra_reci_coup_g6

    def orient_reci_coup_mat_list(self):
        if self.stack_label == "ada":
            # for g2
            orient_reci_coup_g2 = conj(self.orient_reci_coup_g1).T

            # for g3
            orient_reci_coup_g3 = self.orient_reci_coup_g1.copy() * \
                array([
                    [1, exp(1j * 2 * pi / 3)],
                    [exp(1j * 2 * pi / 3), 1]
                ])

            # for g4
            orient_reci_coup_g4 = self.orient_reci_coup_g1.copy() * \
                array([
                    [1, exp(-1j * 2 * pi / 3)],
                    [exp(-1j * 2 * pi / 3), 1]
                ])
            np.fill_diagonal(
                orient_reci_coup_g4, conj(
                    np.diag(
                        self.orient_reci_coup_g1)))

            # for g5
            orient_reci_coup_g5 = conj(orient_reci_coup_g4).T

            # for g6
            orient_reci_coup_g6 = conj(orient_reci_coup_g3).T

            return self.orient_reci_coup_g1, orient_reci_coup_g2, orient_reci_coup_g3, orient_reci_coup_g4, orient_reci_coup_g5, orient_reci_coup_g6
        elif self.stack_label == "adb":
            # for g2
            orient_reci_coup_g2 = conj(self.orient_reci_coup_g1).T

            # for g3
            orient_reci_coup_g3 = self.orient_reci_coup_g1.copy() * \
                array([
                    [exp(1j * 2 * pi / 3), 1],
                    [exp(1j * 2 * pi / 3), exp(1j * 2 * pi / 3)]
                ])

            # for g4
            orient_reci_coup_g4 = self.orient_reci_coup_g1.copy() * \
                array([
                    [1, 1],
                    [1, exp(-1j * 2 * pi / 3)]
                ])
            orient_reci_coup_g4[0][1] = conj(self.orient_reci_coup_g1[0][1])

            # for g5
            orient_reci_coup_g5 = conj(orient_reci_coup_g4).T

            # for g6
            orient_reci_coup_g6 = conj(orient_reci_coup_g3).T

            return self.orient_reci_coup_g1, orient_reci_coup_g2, orient_reci_coup_g3, orient_reci_coup_g4, orient_reci_coup_g5, orient_reci_coup_g6
        else:
            orient_reci_coup_g2 = conj(self.orient_reci_coup_g1).T
            orient_reci_coup_g3 = self.orient_reci_coup_g1.copy()
            orient_reci_coup_g4 = conj(self.orient_reci_coup_g1).T
            orient_reci_coup_g5 = self.orient_reci_coup_g1.copy()
            orient_reci_coup_g6 = conj(self.orient_reci_coup_g1).T

            return self.orient_reci_coup_g1, orient_reci_coup_g2, orient_reci_coup_g3, orient_reci_coup_g4, orient_reci_coup_g5, orient_reci_coup_g6

    @staticmethod
    def layer1to2_multi_coupling(vec, targeted_layer_index):
        first_shell = [
            (vec[0],
             vec[1],
                targeted_layer_index),
            (vec[0] + 1,
             vec[1],
                targeted_layer_index),
            (vec[0],
             vec[1] + 1,
                targeted_layer_index)]
        second_shell = [(vec[0] +
                         1, vec[1] +
                         1, targeted_layer_index), (vec[0] +
                                                    1, vec[1] -
                                                    1, targeted_layer_index), (vec[0] -
                                                                               1, vec[1] +
                                                                               1, targeted_layer_index)]
        third_shell = [(vec[0], vec[1] +
                        2, targeted_layer_index), (vec[0] +
                                                   2, vec[1], targeted_layer_index), (vec[0] -
                                                                                      1, vec[1] +
                                                                                      2, targeted_layer_index), (vec[0] +
                                                                                                                 2, vec[1] -
                                                                                                                 1, targeted_layer_index), (vec[0] -
                                                                                                                                            1, vec[1], targeted_layer_index), (vec[0], vec[1] -
                                                                                                                                                                               1, targeted_layer_index)]
        out_vec_list = []
        out_vec_list.extend(first_shell)
        out_vec_list.extend(second_shell)
        out_vec_list.extend(third_shell)
        return first_shell, second_shell, third_shell, out_vec_list

    @staticmethod
    # counterpart of layer1to2_multi_coupling
    def layer2to1_multi_coupling(vec, targeted_layer_index):
        first_shell = [
            (vec[0],
             vec[1],
                targeted_layer_index),
            (vec[0] - 1,
             vec[1],
                targeted_layer_index),
            (vec[0],
             vec[1] - 1,
                targeted_layer_index)]
        second_shell = [(vec[0] -
                         1, vec[1] -
                         1, targeted_layer_index), (vec[0] -
                                                    1, vec[1] +
                                                    1, targeted_layer_index), (vec[0] +
                                                                               1, vec[1] -
                                                                               1, targeted_layer_index)]
        third_shell = [(vec[0], vec[1] -
                        2, targeted_layer_index), (vec[0] -
                                                   2, vec[1], targeted_layer_index), (vec[0] +
                                                                                      1, vec[1] -
                                                                                      2, targeted_layer_index), (vec[0] -
                                                                                                                 2, vec[1] +
                                                                                                                 1, targeted_layer_index), (vec[0] +
                                                                                                                                            1, vec[1], targeted_layer_index), (vec[0], vec[1] +
                                                                                                                                                                               1, targeted_layer_index)]
        out_vec_list = []
        out_vec_list.extend(first_shell)
        out_vec_list.extend(second_shell)
        out_vec_list.extend(third_shell)
        return first_shell, second_shell, third_shell, out_vec_list

    @staticmethod
    def intra_l_reci_coup_vec(vec, targeted_layer_index):
        vec_1 = (vec[0], vec[1], targeted_layer_index, 0)
        vec_2 = (vec[0], vec[1], targeted_layer_index, 1)
        vec_3 = (vec[0], vec[1], targeted_layer_index, 2)
        vec_4 = (vec[0], vec[1], targeted_layer_index, 3)
        vec_5 = (vec[0], vec[1], targeted_layer_index, 4)
        vec_6 = (vec[0], vec[1], targeted_layer_index, 5)
        return [vec_1, vec_2, vec_3, vec_4, vec_5, vec_6]

    def basis_set(self, loop_times):
        original_basis_list = [(0, 0, 1)]
        times = 0
        vec_list_f = [(0, 0, 1)]
        vec_intra_coupling_dic = {}
        while times < loop_times:
            for ele_vec in original_basis_list:
                if ele_vec[2] == 1:
                    vec_intra_coupling_dic[ele_vec] = self.intra_l_reci_coup_vec(
                        ele_vec, 1)
                    shell1, shell2, shell3, all_inter_shell = self.layer1to2_multi_coupling(
                        ele_vec, self.twist_l_index)
                    for sub_vec in all_inter_shell:
                        if sub_vec in vec_list_f:
                            pass
                        else:
                            vec_intra_coupling_dic[sub_vec] = self.intra_l_reci_coup_vec(
                                sub_vec, self.twist_l_index)
                            vec_list_f.append(sub_vec)
            times = times + 1
            original_basis_list = vec_list_f[:]
            if times < loop_times:
                for ele_vec in original_basis_list:
                    if ele_vec[2] == self.twist_l_index:
                        vec_intra_coupling_dic[ele_vec] = self.intra_l_reci_coup_vec(
                            ele_vec, self.twist_l_index)
                        shell1, shell2, shell3, all_inter_shell = self.layer2to1_multi_coupling(
                            ele_vec, 1)
                        for sub_vec in all_inter_shell:
                            if sub_vec in vec_list_f:
                                pass
                            else:
                                vec_intra_coupling_dic[sub_vec] = self.intra_l_reci_coup_vec(
                                    sub_vec, 1)
                                vec_list_f.append(sub_vec)
            times = times + 1
            original_basis_list = vec_list_f[:]
        for ele_vec in [
                vector for vector in original_basis_list if vector[2] == 1]:
            residual_basis = (ele_vec[0], ele_vec[1], self.res_l_index)
            if residual_basis in vec_list_f:
                pass
            else:
                vec_intra_coupling_dic[residual_basis] = self.intra_l_reci_coup_vec(
                    ele_vec, self.res_l_index)
                vec_list_f.append((ele_vec[0], ele_vec[1], self.res_l_index))
        vec_including_intra_list = []
        for vec_list_values in vec_intra_coupling_dic.values():
            vec_including_intra_list.extend(vec_list_values)

        return vec_list_f  # , vec_including_intra_list

    @staticmethod
    def t_mn(m, n, inter_coup_mat):
        # m n the number of the vector b_p and b_n, for starting from AA stacking, m corresponds to b_p, n corresponds to
        # b_n, so make sure that the number of b_n must be n
        return inter_coup_mat * array(
            [
                [1, exp(-1j * (m - n) * 2 * pi / 3)],
                [exp(1j * (m - n) * 2 * pi / 3), 1]
            ]
        )

    def twist_coup_near(self, origin_bas, coup_bas):
        m = coup_bas[0] - origin_bas[0]
        n = coup_bas[1] - origin_bas[1]
        t_shell1 = self.t_mn(m, n, self.inter_coup_mat_near1)
        t_shell2 = self.t_mn(m, n, self.inter_coup_mat_near2)
        t_shell3 = self.t_mn(m, n, self.inter_coup_mat_near3)
        return t_shell1, t_shell2, t_shell3

    def twist_coup_far(self, origin_bas, coup_bas):
        m = coup_bas[0] - origin_bas[0]
        n = coup_bas[1] - origin_bas[1]
        t_shell1 = self.t_mn(m, n, self.inter_coup_mat_far1)
        t_shell2 = self.t_mn(m, n, self.inter_coup_mat_far2)
        t_shell3 = self.t_mn(m, n, self.inter_coup_mat_far3)
        return t_shell1, t_shell2, t_shell3

    def h_b(self, k_arr):
        base_res_k_x = k_arr[0] - self.k_b_arr[0]
        base_res_k_y = k_arr[1] - self.k_b_arr[1]
        mat_base = self.norm_Kg_conti * self.base_intra_coup * \
            PubMeth.get_k_mat(self.a0_constant_conti, self.twist_theta_conti, self.norm_Kg_conti, base_res_k_x, base_res_k_y)
        mat_res = self.norm_Kg_conti * self.res_intra_coup * \
            PubMeth.get_k_mat(self.a0_constant_conti, self.twist_theta_conti, self.norm_Kg_conti, base_res_k_x, base_res_k_y)
        return mat_base, mat_res

    def h_t(self, k_arr):
        twist_k_x = k_arr[0] - self.k_t_arr[0]
        twist_k_y = k_arr[1] - self.k_t_arr[1]
        return self.norm_Kg_conti * self.twist_intra_coup * \
            PubMeth.get_k_mat(self.a0_constant_conti, self.twist_theta_conti, self.norm_Kg_conti, twist_k_x, twist_k_y)

    def hamiltonian_construction(self, k):
        h_out = []
        for bra_vec in self.pre_basis_list:
            if bra_vec[2] == self.base_l_index:
                h_r = []
                shell1, shell2, shell3, all_shell = self.layer1to2_multi_coupling(
                    bra_vec, self.twist_l_index)
                for ket_vec in self.pre_basis_list:
                    # self hamiltonian
                    if ket_vec == bra_vec:
                        h_r.append(
                            self.h_b(
                                k +
                                ket_vec[0] *
                                self.b_p_arr +
                                ket_vec[1] *
                                self.b_n_arr)[0])
                    # oriented layer coupling
                    elif ket_vec == (bra_vec[0], bra_vec[1], self.res_l_index):
                        h_r.append(self.orient_coup)
                    # twisted layer coupling
                    elif ket_vec in shell1:
                        if abs(self.twist_l_index - bra_vec[2]) == 1:
                            h_r.append(
                                self.twist_coup_near(
                                    bra_vec, ket_vec)[0])
                        else:
                            h_r.append(
                                self.twist_coup_far(
                                    bra_vec, ket_vec)[0])
                    elif ket_vec in shell2:
                        if abs(self.twist_l_index - bra_vec[2]) == 1:
                            h_r.append(
                                self.twist_coup_near(
                                    bra_vec, ket_vec)[1])
                        else:
                            h_r.append(
                                self.twist_coup_far(
                                    bra_vec, ket_vec)[1])
                    elif ket_vec in shell3:
                        if abs(self.twist_l_index - bra_vec[2]) == 1:
                            h_r.append(
                                self.twist_coup_near(
                                    bra_vec, ket_vec)[2])
                        else:
                            h_r.append(
                                self.twist_coup_far(
                                    bra_vec, ket_vec)[2])
                    # intra-layer reci coupling
                    elif (ket_vec[0] - bra_vec[0], ket_vec[1] - bra_vec[1]) in self.moire_reci_vecs and ket_vec[2] == bra_vec[2]:
                        h_r.append(
                            self.intra_reci_coup_mat_list(
                                self.intra_reci_coup_g1_base)[
                                self.which_reci(
                                    bra_vec, ket_vec)])
                    # oriented reci coupling
                    elif (ket_vec[0] - bra_vec[0], ket_vec[1] - bra_vec[1]) in self.moire_reci_vecs and ket_vec[2] == self.res_l_index:
                        h_r.append(
                            self.orient_reci_coup_mat_list()[
                                self.which_reci(
                                    bra_vec, ket_vec)])

                    else:
                        h_r.append(np.zeros((2, 2)))
                h_out.append(h_r)
            elif bra_vec[2] == self.res_l_index:
                h_r = []
                shell1, shell2, shell3, all_shell = self.layer1to2_multi_coupling(
                    bra_vec, self.twist_l_index)
                for ket_vec in self.pre_basis_list:
                    # self hamiltonian
                    if ket_vec == bra_vec:
                        h_r.append(
                            self.h_b(
                                k +
                                ket_vec[0] *
                                self.b_p_arr +
                                ket_vec[1] *
                                self.b_n_arr)[1])
                    # oriented layer coupling
                    elif ket_vec == (bra_vec[0], bra_vec[1], self.base_l_index):
                        h_r.append(conj(self.orient_coup).T)
                    # intra reci coupling

                    # twisted layer coupling
                    elif ket_vec in shell1:
                        if abs(self.twist_l_index - bra_vec[2]) == 1:
                            h_r.append(
                                self.twist_coup_near(
                                    bra_vec, ket_vec)[0])
                        else:
                            h_r.append(
                                self.twist_coup_far(
                                    bra_vec, ket_vec)[0])
                    elif ket_vec in shell2:
                        if abs(self.twist_l_index - bra_vec[2]) == 1:
                            h_r.append(
                                self.twist_coup_near(
                                    bra_vec, ket_vec)[1])
                        else:
                            h_r.append(
                                self.twist_coup_far(
                                    bra_vec, ket_vec)[1])
                    elif ket_vec in shell3:
                        if abs(self.twist_l_index - bra_vec[2]) == 1:
                            h_r.append(
                                self.twist_coup_near(
                                    bra_vec, ket_vec)[2])
                        else:
                            h_r.append(
                                self.twist_coup_far(
                                    bra_vec, ket_vec)[2])
                    # intra-layer reci coupling
                    elif (ket_vec[0] - bra_vec[0], ket_vec[1] - bra_vec[1]) in self.moire_reci_vecs and ket_vec[2] == bra_vec[2]:
                        h_r.append(
                            self.intra_reci_coup_mat_list(
                                self.intra_reci_coup_g1_res)[
                                self.which_reci(
                                    bra_vec, ket_vec)])
                    # oriented reci coupling
                    elif (ket_vec[0] - bra_vec[0], ket_vec[1] - bra_vec[1]) in self.moire_reci_vecs and ket_vec[2] == self.base_l_index:
                        h_r.append(
                            self.orient_reci_coup_mat_list()[
                                self.which_reci(
                                    bra_vec, ket_vec)])
                    else:
                        h_r.append(np.zeros((2, 2)))
                h_out.append(h_r)
            elif bra_vec[2] == self.twist_l_index:
                h_r = []
                shell1_1, shell2_1, shell3_1, all_shell_1 = self.layer2to1_multi_coupling(
                    bra_vec, self.base_l_index)
                shell1_2, shell2_2, shell3_2, all_shell_2 = self.layer2to1_multi_coupling(
                    bra_vec, self.res_l_index)
                for ket_vec in self.pre_basis_list:
                    # self hamiltonian
                    if ket_vec == bra_vec:
                        h_r.append(
                            self.h_t(
                                k +
                                ket_vec[0] *
                                self.b_p_arr +
                                ket_vec[1] *
                                self.b_n_arr))
                    # twisted layer coupling
                    elif ket_vec in shell1_1:
                        if abs(self.base_l_index - bra_vec[2]) == 1:
                            h_r.append(
                                conj(
                                    self.twist_coup_near(
                                        ket_vec,
                                        bra_vec)[0]).T)
                        else:
                            h_r.append(
                                conj(
                                    self.twist_coup_far(
                                        ket_vec,
                                        bra_vec)[0]).T)
                    elif ket_vec in shell2_1:
                        if abs(self.base_l_index - bra_vec[2]) == 1:
                            h_r.append(
                                conj(
                                    self.twist_coup_near(
                                        ket_vec,
                                        bra_vec)[1]).T)
                        else:
                            h_r.append(
                                conj(
                                    self.twist_coup_far(
                                        ket_vec,
                                        bra_vec)[1]).T)
                    elif ket_vec in shell3_1:
                        if abs(self.base_l_index - bra_vec[2]) == 1:
                            h_r.append(
                                conj(
                                    self.twist_coup_near(
                                        ket_vec,
                                        bra_vec)[2]).T)
                        else:
                            h_r.append(
                                conj(
                                    self.twist_coup_far(
                                        ket_vec,
                                        bra_vec)[2]).T)
                    elif ket_vec in shell1_2:
                        if abs(self.res_l_index - bra_vec[2]) == 1:
                            h_r.append(
                                conj(
                                    self.twist_coup_near(
                                        ket_vec,
                                        bra_vec)[0]).T)
                        else:
                            h_r.append(
                                conj(
                                    self.twist_coup_far(
                                        ket_vec,
                                        bra_vec)[0]).T)
                    elif ket_vec in shell2_2:
                        if abs(self.res_l_index - bra_vec[2]) == 1:
                            h_r.append(
                                conj(
                                    self.twist_coup_near(
                                        ket_vec,
                                        bra_vec)[1]).T)
                        else:
                            h_r.append(
                                conj(
                                    self.twist_coup_far(
                                        ket_vec,
                                        bra_vec)[1]).T)
                    elif ket_vec in shell3_2:
                        if abs(self.res_l_index - bra_vec[2]) == 1:
                            h_r.append(
                                conj(
                                    self.twist_coup_near(
                                        ket_vec,
                                        bra_vec)[2]).T)
                        else:
                            h_r.append(
                                conj(
                                    self.twist_coup_far(
                                        ket_vec,
                                        bra_vec)[2]).T)
                    # intra-layer reci coupling
                    elif (ket_vec[0] - bra_vec[0], ket_vec[1] - bra_vec[1]) in self.moire_reci_vecs and ket_vec[2] == bra_vec[2]:
                        h_r.append(
                            self.intra_reci_coup_mat_list(
                                self.intra_reci_coup_g1_twist)[
                                self.which_reci(
                                    bra_vec, ket_vec)])
                    # no oriented reci coupling
                    else:
                        h_r.append(np.zeros((2, 2)))
                h_out.append(h_r)
        return np.block(h_out)


class ContiAadTtgInst(ContiTtgInst):

    base_intra_coup = PubMeth.list2mat([6, 2611, 2611, 6])

    res_intra_coup = PubMeth.list2mat([-9, 2606, 2606, -9])

    twist_intra_coup = PubMeth.list2mat([3, 2605, 2605, 3])

    inter_coup_mat_near1 = PubMeth.list2mat([91, 106, 106, 91])

    inter_coup_mat_near2 = PubMeth.list2mat([-9, 6, 6, -9])

    inter_coup_mat_near3 = PubMeth.list2mat(
        [-5, 4 * exp(1j * 47 / 180 * pi), 4 * exp(1j * 47 / 180 * pi), -5])

    inter_coup_mat_far1 = PubMeth.list2mat([2, 3, 3, 2])

    inter_coup_mat_far2 = 0
    inter_coup_mat_far3 = 0

    orient_coup = PubMeth.list2mat([225, 0, 0, 225])

    intra_reci_coup_g1_base = zeros((2, 2))
    intra_reci_coup_g1_res = PubMeth.putin2mat(
        [(2, 28), (2, 60), (2, 60), (2, 28)])
    intra_reci_coup_g1_twist = PubMeth.putin2mat(
        [(2, -31), (2, 60), (2, 60), (2, -31)])
    orient_reci_coup_g1 = zeros((2, 2))

    def __init__(self, twist_angle):
        super().__init__(
            twist_angle,
            twist_l_index=3,
            res_l_index=2,
            base_intra_coup=self.base_intra_coup,
            res_intra_coup=self.res_intra_coup,
            twist_intra_coup=self.twist_intra_coup,
            inter_coup_mat_near1=self.inter_coup_mat_near1,
            inter_coup_mat_near2=self.inter_coup_mat_near2,
            inter_coup_mat_near3=self.inter_coup_mat_near3,
            inter_coup_mat_far1=self.inter_coup_mat_far1,
            inter_coup_mat_far2=self.inter_coup_mat_far2,
            inter_coup_mat_far3=self.inter_coup_mat_far3,
            orient_coup=self.orient_coup,
            orient_reci_coup_g1=self.orient_reci_coup_g1,
            intra_reci_coup_g1_base=self.intra_reci_coup_g1_base,
            intra_reci_coup_g1_res=self.intra_reci_coup_g1_res,
            intra_reci_coup_g1_twist=self.intra_reci_coup_g1_twist,
            stack_label='aad')


class ContiAbdTtgInst(ContiTtgInst):

    base_intra_coup = PubMeth.list2mat([2, 2591, 2591, 2])

    res_intra_coup = PubMeth.list2mat([-13, 2608, 2608, -13])

    twist_intra_coup = PubMeth.list2mat([11, 2606, 2606, 11])

    inter_coup_mat_near1 = PubMeth.list2mat([90, 105, 105, 90])

    inter_coup_mat_near2 = PubMeth.list2mat([-9, 5, 5, -9])

    inter_coup_mat_near3 = PubMeth.list2mat(
        [-4, 5 * exp(1j * 56 / 180 * pi), 5 * exp(1j * 56 / 180 * pi), -4])

    inter_coup_mat_far1 = PubMeth.list2mat([3, 3, 3, 3])

    inter_coup_mat_far2 = 0
    inter_coup_mat_far3 = 0

    orient_coup = PubMeth.list2mat([0, 357, 0, 0])

    intra_reci_coup_g1_base = zeros((2, 2))
    intra_reci_coup_g1_res = PubMeth.putin2mat(
        [(2, -40), (2, 60), (2, 60), (2, -40)])
    intra_reci_coup_g1_twist = PubMeth.putin2mat(
        [(2, 30), (2, 60), (2, 60), (2, 30)])
    orient_reci_coup_g1 = array([[0, 2], [0, 0]])

    def __init__(self, twist_angle, raman_gamma=100, kp_num=70, basis_loop_times=3):
        super().__init__(
            twist_angle,
            twist_l_index=3,
            res_l_index=2,
            base_intra_coup=self.base_intra_coup,
            res_intra_coup=self.res_intra_coup,
            twist_intra_coup=self.twist_intra_coup,
            inter_coup_mat_near1=self.inter_coup_mat_near1,
            inter_coup_mat_near2=self.inter_coup_mat_near2,
            inter_coup_mat_near3=self.inter_coup_mat_near3,
            inter_coup_mat_far1=self.inter_coup_mat_far1,
            inter_coup_mat_far2=self.inter_coup_mat_far2,
            inter_coup_mat_far3=self.inter_coup_mat_far3,
            orient_coup=self.orient_coup,
            orient_reci_coup_g1=self.orient_reci_coup_g1,
            intra_reci_coup_g1_base=self.intra_reci_coup_g1_base,
            intra_reci_coup_g1_res=self.intra_reci_coup_g1_res,
            intra_reci_coup_g1_twist=self.intra_reci_coup_g1_twist,
            stack_label='abd',
            kp_num=kp_num,
            raman_gamma=raman_gamma,
            basis_loop_times=basis_loop_times)


class ContiAdaTtgInst(ContiTtgInst):

    base_intra_coup = PubMeth.list2mat([5, 2605, 2605, 5])

    res_intra_coup = PubMeth.list2mat([5, 2605, 2605, 5])

    twist_intra_coup = PubMeth.list2mat([-10, 2602, 2602, -10])

    inter_coup_mat_near1 = PubMeth.list2mat([93, 105, 105, 93])

    inter_coup_mat_near2 = PubMeth.list2mat([-8, 4, 4, -8])

    inter_coup_mat_near3 = PubMeth.list2mat(
        [-5, 4 * exp(1j * 50), 4 * exp(1j * 50), -5])

    inter_coup_mat_far1 = PubMeth.list2mat([93, 105, 105, 93])

    inter_coup_mat_far2 = PubMeth.list2mat([-8, 4, 4, -8])

    inter_coup_mat_far3 = PubMeth.list2mat(
        [-5, 4 * exp(1j * 50), 4 * exp(1j * 50), -5])

    orient_coup = PubMeth.list2mat([4, 0, 0, 4])

    intra_reci_coup_g1_base = PubMeth.putin2mat(
        [(2, -30), (2, 60), (2, 60), (2, -30)])
    intra_reci_coup_g1_res = PubMeth.putin2mat(
        [(3, 30), (3, 60), (3, 60), (3, 30)])
    intra_reci_coup_g1_twist = PubMeth.putin2mat(
        [(2, -30), (2, 60), (2, 60), (2, -30)])
    orient_reci_coup_g1 = PubMeth.putin2mat(
        [(2, -70), (2, 60), (2, -60), (2, 70)])

    def __init__(self, twist_angle):
        super().__init__(
            twist_angle,
            twist_l_index=2,
            res_l_index=3,
            base_intra_coup=self.base_intra_coup,
            res_intra_coup=self.res_intra_coup,
            twist_intra_coup=self.twist_intra_coup,
            inter_coup_mat_near1=self.inter_coup_mat_near1,
            inter_coup_mat_near2=self.inter_coup_mat_near2,
            inter_coup_mat_near3=self.inter_coup_mat_near3,
            inter_coup_mat_far1=self.inter_coup_mat_far1,
            inter_coup_mat_far2=self.inter_coup_mat_far2,
            inter_coup_mat_far3=self.inter_coup_mat_far3,
            orient_coup=self.orient_coup,
            orient_reci_coup_g1=self.orient_reci_coup_g1,
            intra_reci_coup_g1_base=self.intra_reci_coup_g1_base,
            intra_reci_coup_g1_res=self.intra_reci_coup_g1_res,
            intra_reci_coup_g1_twist=self.intra_reci_coup_g1_twist,
            stack_label='ada')


class ContiAdbTtgInst(ContiTtgInst):

    base_intra_coup = PubMeth.list2mat([5, 2605, 2605, 5])

    res_intra_coup = PubMeth.list2mat([6, 2583, 2583, 6])

    twist_intra_coup = PubMeth.list2mat([-11, 2602, 2602, -11])

    inter_coup_mat_near1 = PubMeth.list2mat([94, 107, 107, 94])

    inter_coup_mat_near2 = PubMeth.list2mat([-8, 6, 6, -8])

    inter_coup_mat_near3 = PubMeth.list2mat(
        [-4, 5 * exp(1j * 49 / 180 * pi), 5 * exp(1j * 49 / 180 * pi), -4])

    inter_coup_mat_far1 = PubMeth.list2mat([91, 105, 105, 91])

    inter_coup_mat_far2 = PubMeth.list2mat([-9, 5, 5, -9])

    inter_coup_mat_far3 = PubMeth.list2mat(
        [-4, 5 * exp(1j * 43 / 180 * pi), 5 * exp(1j * 43 / 180 * pi), -4])

    orient_coup = PubMeth.list2mat([0, 4, 0, 0])

    intra_reci_coup_g1_base = PubMeth.putin2mat(
        [(1, -60), (2, 60), (2, 60), (1, -60)])
    intra_reci_coup_g1_res = PubMeth.putin2mat(
        [(2, -36), (1, 0), (1, 0), (2, -36)])
    intra_reci_coup_g1_twist = PubMeth.putin2mat(
        [(1, -150), (2, -60), (2, -60), (1, -150)])
    orient_reci_coup_g1 = PubMeth.putin2mat(
        [(-2, 0), (2, -60), (-2, 0), (2, -60)])

    def __init__(self, twist_angle):
        super().__init__(
            twist_angle,
            twist_l_index=2,
            res_l_index=3,
            base_intra_coup=self.base_intra_coup,
            res_intra_coup=self.res_intra_coup,
            twist_intra_coup=self.twist_intra_coup,
            inter_coup_mat_near1=self.inter_coup_mat_near1,
            inter_coup_mat_near2=self.inter_coup_mat_near2,
            inter_coup_mat_near3=self.inter_coup_mat_near3,
            inter_coup_mat_far1=self.inter_coup_mat_far1,
            inter_coup_mat_far2=self.inter_coup_mat_far2,
            inter_coup_mat_far3=self.inter_coup_mat_far3,
            orient_coup=self.orient_coup,
            orient_reci_coup_g1=self.orient_reci_coup_g1,
            intra_reci_coup_g1_base=self.intra_reci_coup_g1_base,
            intra_reci_coup_g1_res=self.intra_reci_coup_g1_res,
            intra_reci_coup_g1_twist=self.intra_reci_coup_g1_twist,
            stack_label='adb')


class TightTtgInst(TightTbgInst):

    def __init__(
            self,
            m0,
            r,
            twist_l_index,
            res_l_index,
            stack_shift,
            base_l_index=1):
        self.twist_l_index = twist_l_index
        self.res_l_index = res_l_index
        self.stack_shift = stack_shift
        self.base_l_index = base_l_index
        super().__init__(m0, r)
        if self.r % 3 != 0:
            self.n_atoms = int(6 * (3 * m0 ** 2 + 3 * m0 * r + r ** 2))
        else:
            self.n_atoms = int(6 * (m0 ** 2 + m0 * r + r ** 2 / 3))
        self.mat_type = 'TTG'

    def atom_positions(self, lattice_index_list):
        # sublattices, stacking_shift==0 corresponds to AA stacking, ==1
        # corresponds to AB stacking
        bound_list_f = self.arrange_bound_p()
        atoms_within = []
        for ele_index in lattice_index_list:
            A1 = ele_index[0] * self.a1 + ele_index[1] * self.a2
            B1 = ele_index[0] * self.a1 + ele_index[1] * self.a2 + self.delta
            A_rot = ele_index[0] * self.r_a1 + ele_index[1] * self.r_a2
            B_rot = ele_index[0] * self.r_a1 + \
                ele_index[1] * self.r_a2 + self.r_delta
            A_res = ele_index[0] * self.a1 + ele_index[1] * \
                self.a2 + self.stack_shift * self.delta
            B_res = ele_index[0] * self.a1 + ele_index[1] * \
                self.a2 + self.delta + self.stack_shift * self.delta
            if PubMeth.isInterArea(
                    A1, bound_list_f) and not PubMeth.at_corners(
                    A1, bound_list_f):
                atoms_within.append((A1[0], A1[1], self.base_l_index))
            if PubMeth.isInterArea(
                    B1, bound_list_f) and not PubMeth.at_corners(
                    B1, bound_list_f):
                atoms_within.append((B1[0], B1[1], self.base_l_index))
            if PubMeth.isInterArea(
                    A_rot,
                    bound_list_f) and not PubMeth.at_corners(
                    A_rot,
                    bound_list_f):
                atoms_within.append((A_rot[0], A_rot[1], self.twist_l_index))
            if PubMeth.isInterArea(
                    B_rot,
                    bound_list_f) and not PubMeth.at_corners(
                    B_rot,
                    bound_list_f):
                atoms_within.append((B_rot[0], B_rot[1], self.twist_l_index))
            if PubMeth.isInterArea(
                    A_res,
                    bound_list_f) and not PubMeth.at_corners(
                    A_res,
                    bound_list_f):
                atoms_within.append((A_res[0], A_res[1], self.res_l_index))
            if PubMeth.isInterArea(
                    B_res,
                    bound_list_f) and not PubMeth.at_corners(
                    B_res,
                    bound_list_f):
                atoms_within.append((B_res[0], B_res[1], self.res_l_index))
        atoms_within.append((0, 0, self.base_l_index))
        atoms_within.append((0, 0, self.twist_l_index))
        if self.stack_shift == 0 or self.stack_shift == 2:
            atoms_within.append((0, 0, self.res_l_index))
        else:
            pass
        return atoms_within

    def plot_atoms(self, point_index_list, save_or_not=False, show_or_not=False):
        all_positions = []
        for ele_index in point_index_list:
            all_positions.append(
                ele_index[0] *
                self.a1 +
                ele_index[1] *
                self.a2)
            all_positions.append(
                ele_index[0] *
                self.a1 +
                ele_index[1] *
                self.a2 +
                self.delta)
            all_positions.append(
                ele_index[0] *
                self.r_a1 +
                ele_index[1] *
                self.r_a2)
            all_positions.append(
                ele_index[0] *
                self.r_a1 +
                ele_index[1] *
                self.r_a2 +
                self.r_delta)
            all_positions.append(
                ele_index[0] *
                self.a1 +
                ele_index[1] *
                self.a2 + self.stack_shift * self.delta)
            all_positions.append(
                ele_index[0] *
                self.a1 +
                ele_index[1] *
                self.a2 +
                self.delta + self.stack_shift * self.delta)

        cor_x = array(all_positions)[:, 0]
        cor_y = array(all_positions)[:, 1]
        bound = self.arrange_bound_p()
        bound.append((0, 0))

        plt.scatter(cor_x, cor_y, marker='.')

        plt.plot(array(bound)[:, 0], array(bound)[:, 1])

        ax = plt.gca()
        ax.set_aspect('equal')
        plt.title(r"Atomic Structure of Trilayer Graphene $\theta=%.2f \degree$" % self.twist_angle, fontsize=14)
        plt.xlabel("X", fontsize=12)
        plt.ylabel("Y", fontsize=12)
        save_path =  PubMeth.get_right_save_path('tmp_figs')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if save_or_not:
            plt.savefig(save_path + 'Atomic_structure_{:.2f}{}.png'.format(self.twist_angle, self.mat_type), dpi=300)
        if show_or_not:
            plt.show()

    def inter_term(self, k_arr, distance_arr, layer1_i, layer2_i):
        d_inter = self.d0 * abs(layer1_i - layer2_i)
        d_norm = sqrt(norm(distance_arr) ** 2 + d_inter ** 2)
        value_cos = d_inter / d_norm
        return (self.t_intra * exp(- (d_norm - self.a0 / sqrt(3)) / self.delta_0_par) * (1 - value_cos ** 2) + self.t_inter *
                exp(- (d_norm - self.d0) / self.delta_0_par) * value_cos ** 2) * exp(-1j * dot(k_arr, distance_arr))

    def hamiltonian_construction(
            self,
            k_arr_f,
            all_p_relations_dict,
            points_tuple_list,
    ):
        # print("this tuples: ", points_tuple_list)
        h_mat = []
        for bra_p_f in points_tuple_list:
            row = []
            relation_dict_f = all_p_relations_dict[bra_p_f]
            for ket_p_f in points_tuple_list:
                if ket_p_f == bra_p_f:
                    row.append(0)
                elif ket_p_f[2] == bra_p_f[2]:
                    row.append(
                        self.intra_term(
                            k_arr_f,
                            relation_dict_f[ket_p_f]))
                elif ket_p_f[2] != bra_p_f[2]:
                    row.append(
                        self.inter_term(
                            k_arr_f,
                            relation_dict_f[ket_p_f],
                            bra_p_f[2],
                            ket_p_f[2]))
            h_mat.append(row)
        return array(h_mat)

    def test(self, all_p_rels, all_tuples):
        self.hamiltonian_construction(array([1, 2]), all_p_rels, all_tuples)


def main():

    # aad = TightTtgInst(1, 1, 3, 2, stack_shift=1)
    # point_indices = aad.proper_lattice_indices()
    # all_tuples = aad.atom_positions(point_indices)
    # all_p_relations = aad.all_relation_dict(all_tuples)
    # path = [aad.K_2, aad.path_gamma, aad.M, aad.K_1]
    # # aad.multi_proc_path(path, all_p_relations, all_tuples, [-2, 2], show=True)

    abd = ContiAbdTtgInst(12.3, basis_loop_times=4)
    # vec_list = abd.basis_set(3)
    # print("kp num: ", abd.kp_num)
    # print("len of vec: ", len(vec_list))
    # e_list = abd.multi_proc_path(path, vec_list)
    # plt.plot(e_list)
    # plt.show()
    # abd.plot_along_path(vec_list, 0, 0, [-2000, 2000], "k", show=True)
    # print(aad.n_atoms)
    # print(path)

    # abd = ContiAbdTtgInst(12.3)
    # vec_list = abd.basis_set(3)
    # e_photon_list = linspace(100, 5000, 50)
    # raman_i_list = []
    # for ele_photon in e_photon_list:
    #     args_list = [20, vec_list, ele_photon]
    #     tmp_i = abd.multi_proc_raman_i_cal(args_list)
    #     raman_i_list.append(tmp_i)
    # np.save("ttg_raman_i.npy", raman_i_list)
    # plt.plot(e_photon_list, raman_i_list)
    # plt.xlabel("E(meV)")
    # plt.ylabel("Raman Intensity")
    # plt.title(r"Raman Intensity of 12.3$\degree$-TTG")
    # plt.savefig("ttg_raman_i.png")
    # plt.close()

    # tb_abd = TightTtgInst(1, 1, 3, 2, 1)
    # indices = tb_abd.proper_lattice_indices()
    # all_tuples = tb_abd.atom_positions(indices)
    # all_relations = tb_abd.all_relation_dict(all_tuples)
    # raman_i_list2 = []
    # for ele_photon in e_photon_list:
    #     args_list2 = [20, ]
    #     tmp_i = tb_ab


    # # # Conti tri raman cal for multi twists
    # raman_i_list = []
    # twist_list = linspace(5, 20, 50)
    # e_photon = 1240 / 532 * 1000  # meV
    # for ele_twist in twist_list:
    #     print("Calculation: 50 angles, 40 half bands, raman_gamma=10meV, kp_num=300 detailed running")
    #     abd = ContiAbdTtgInst(ele_twist, raman_gamma=10, kp_num=300)
    #     vec_list = abd.basis_set(3)
    #     print("len of vecs: ", len(vec_list))
    #     print("# of atoms: ", abd.N_k)
    #     # path = [abd.k_b_p, abd.m_p1, abd.k_t_p, abd.gamma0, abd.gamma1, abd.k_b_p]
    #     args_in = [40, vec_list, e_photon]
    #     tmp_i = abd.multi_proc_raman_2d(args_in, figs_save=True)
    #     raman_i_list.append(tmp_i)
    #     print("Complete twist: ", ele_twist)
    #     print("Now the raman intensity list is: ", raman_i_list)
    # np.save("raman_div_test_40half.npy", raman_i_list)
    # plt.figure(figsize=(7, 4), dpi=300)
    # plt.plot(twist_list, raman_i_list)
    # plt.xlabel(r"$\theta(\degree)$", fontsize=12)
    # plt.ylabel("Raman Intensity", fontsize=12) 
    # plt.title("Raman Intensity at Different Twist Angles", fontsize=14)
    # plt.savefig("raman_div_test_40half.png")
    # plt.close()


if __name__ == "__main__":
    main()