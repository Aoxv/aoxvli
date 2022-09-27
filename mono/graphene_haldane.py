from public.method import *


class HaldaneGra:

    def __init__(self, a0_const = 1.42 * sqrt(3), M=1, t1=1, t2=1, phi_theta=0, density_on_side=100) -> None:
        self.a0_const = a0_const
        self.a1_arr = self.a0_const * array([-1 / 2, sqrt(3) / 2])
        self.a2_arr = self.a0_const * array([1 / 2, sqrt(3) / 2])
        self.b1_arr = 2 * pi / (sqrt(3) * self.a0_const) * array([-sqrt(3), 1])
        self.b2_arr = 2 * pi / (sqrt(3) * self.a0_const) * array([sqrt(3), 1])
        self.delta1_arr = self.a0_const * array([-1 / 2, 1 / (2 * sqrt(3))])
        self.delta2_arr = self.a0_const * array([1 / 2, 1 / (2 * sqrt(3))])
        self.delta3_arr = self.a0_const * array([0, -1 / sqrt(3)])
        self.K_valley_arr = 4 * pi / (3 * self.a0_const) * array([1, 0])
        self.K_valley_prime_arr = 4 * pi / (3 * self.a0_const) * array([-1, 0])
        self.t1 = t1
        self.t2 = t2
        self.M = M
        self.phi_theta = phi_theta
        self.dots_density = density_on_side / norm(self.K_valley_arr)

    def h_x(self, k_arr):
        return -self.t1 * (cos(k_arr @ self.delta1_arr) + cos(k_arr @ self.delta2_arr) + cos(k_arr @ self.delta3_arr))
    
    def h_y(self, k_arr):
        return -self.t1 * (sin(k_arr @ self.delta1_arr) + sin(k_arr @ self.delta2_arr) + sin(k_arr @ self.delta3_arr))
    
    def h_z(self, k_arr):
        return self.M - 2 * self.t2 * sin(self.phi_theta) * (sin(k_arr @ (self.a1_arr - self.a2_arr)) + sin(k_arr @ self.a2_arr) - sin(k_arr @ self.a1_arr))
    
    def h_0(self, k_arr):
        return -2 * self.t2 * cos(self.phi_theta) * (cos(k_arr @ (self.a1_arr - self.a2_arr)) + cos(k_arr @ self.a2_arr) + cos(k_arr @ self.a1_arr))
        
    def hamiltonian(self, k_arr):
        return self.h_0(k_arr) * PubMeth.pauli_mat(0) + self.h_x(k_arr) * PubMeth.pauli_mat(1) + self.h_y(k_arr) * PubMeth.pauli_mat(2) + self.h_z(k_arr) * PubMeth.pauli_mat(3)

    def kps_in_BZ(self, delta_xy):
        out_arr_list = []
        for ele_m in arange(0, 1, delta_xy):
            for ele_n in arange(0, 1, delta_xy):
                out_arr_list.append(ele_n * self.b1_arr + ele_m * self.b2_arr)
        return out_arr_list

    def berry_cur(self, k_arr, delta_kx, delta_ky):
        
        cent_h = self.hamiltonian(k_arr)
        eigen_values, eigen_vectors = np.linalg.eig(cent_h)
        cent_vec = eigen_vectors[:, np.argsort(real(eigen_values))[0]]

        delta_kx_h = self.hamiltonian(k_arr + array([delta_kx, 0]))
        eigen_values, eigen_vectors = np.linalg.eig(delta_kx_h)
        delta_kx_vec = eigen_vectors[:, np.argsort(real(eigen_values))[0]]

        delta_ky_h = self.hamiltonian(k_arr + array([0, delta_ky]))
        eigen_values, eigen_vectors = np.linalg.eig(delta_ky_h)
        delta_ky_vec = eigen_vectors[:, np.argsort(real(eigen_values))[0]]

        delta_kx_ky_h = self.hamiltonian(k_arr + array([delta_kx, delta_ky]))
        eigen_values, eigen_vectors = np.linalg.eig(delta_kx_ky_h)
        delta_kx_ky_vec = eigen_vectors[:, np.argsort(real(eigen_values))[0]]

        U_x = conj(cent_vec) @ delta_kx_vec / abs(conj(cent_vec) @ delta_kx_vec)
        U_y = conj(cent_vec) @ delta_ky_vec / abs(conj(cent_vec) @ delta_ky_vec)
        U_xy = conj(delta_ky_vec) @ delta_kx_ky_vec / abs(conj(delta_ky_vec) @ delta_kx_ky_vec)
        U_yx = conj(delta_kx_vec) @ delta_kx_ky_vec / abs(conj(delta_kx_vec) @ delta_kx_ky_vec)

        F_12 = log(U_x * U_yx / (U_xy * U_y))

        return F_12

    def multi_proc_chern_num(self, args_list, save_2d_plots=True, mat_2d_save=True, cmap='jet'):
        """
        :param args_list: [delta_space]
        :return: chern number
        """
        delta_kx = self.b2_arr[0] * args_list[0] * 2
        delta_ky = self.b2_arr[1] * args_list[0]

        arr_list = self.kps_in_BZ(args_list[0])
        parts_list = PubMeth.divide_list(arr_list)

        berry_cur_list = PubMeth.multi_proc_func(self.berry_cur, parts_list, [delta_kx, delta_ky])
        chern_num = -sum(berry_cur_list) / (1j * 2 * pi)
        if save_2d_plots:
            PubMeth.rect2diam(-array(imag(berry_cur_list)).reshape((int(sqrt(len(berry_cur_list))), int(sqrt(len(berry_cur_list))))), 'Berry_cur', r'$F_{12}$', save_2d_plots=True, rm_raw=True, cmap=cmap, direction_up=False)
        if mat_2d_save:
            np.save('berry_cur.npy', berry_cur_list)
        
        return chern_num
        
    
    # def chern_num(self, delta_space):
    #     chern_num = 0
    #     arr_list = self.kps_in_BZ(delta_space)
    #     delta_kx = self.b2_arr[0] * delta_space * 2
    #     delta_ky = self.b2_arr[1] * delta_space

    #     # plt.scatter([ele[0] for ele in arr_list], [ele[1] for ele in arr_list], marker='.')
    #     # plt.scatter(delta_kx, 0)
    #     # plt.scatter(0, delta_ky)
    #     # ax = plt.gca()
    #     # ax.set_aspect('equal')
    #     # ax.set_xlabel('x', fontsize=12)
    #     # ax.set_ylabel('y', fontsize=12)
    #     # ax.set_title('', fontsize=14)
    #     # plt.xlim(ax.get_xlim())
    #     # plt.ylim(ax.get_ylim())
    #     # plt.savefig('test.png', dpi=300, facecolor='w')
    #     # plt.close()

    #     for ele_arr in arr_list:
    #         cent_h = self.hamiltonian(ele_arr)
    #         eigen_values, eigen_vectors = np.linalg.eig(cent_h)
    #         cent_vec = eigen_vectors[:, np.argsort(real(eigen_values))[0]]

    #         delta_kx_h = self.hamiltonian(ele_arr + array([delta_kx, 0]))
    #         eigen_values, eigen_vectors = np.linalg.eig(delta_kx_h)
    #         delta_kx_vec = eigen_vectors[:, np.argsort(real(eigen_values))[0]]

    #         delta_ky_h = self.hamiltonian(ele_arr + array([0, delta_ky]))
    #         eigen_values, eigen_vectors = np.linalg.eig(delta_ky_h)
    #         delta_ky_vec = eigen_vectors[:, np.argsort(real(eigen_values))[0]]

    #         delta_kx_ky_h = self.hamiltonian(ele_arr + array([delta_kx, delta_ky]))
    #         eigen_values, eigen_vectors = np.linalg.eig(delta_kx_ky_h)
    #         delta_kx_ky_vec = eigen_vectors[:, np.argsort(real(eigen_values))[0]]

    #         U_x = conj(cent_vec) @ delta_kx_vec / abs(conj(cent_vec) @ delta_kx_vec)
    #         U_y = conj(cent_vec) @ delta_ky_vec / abs(conj(cent_vec) @ delta_ky_vec)
    #         U_xy = conj(delta_ky_vec) @ delta_kx_ky_vec / abs(conj(delta_ky_vec) @ delta_kx_ky_vec)
    #         U_yx = conj(delta_kx_vec) @ delta_kx_ky_vec / abs(conj(delta_kx_vec) @ delta_kx_ky_vec)

    #         F_12 = log(U_x * U_yx / (U_xy * U_y))
    #         chern_num = chern_num + F_12
        
    #     chern_num = -chern_num / (1j * 2 * pi)
    #     print("Chern Number: ", chern_num)
    #     return chern_num
