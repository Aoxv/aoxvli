from public.method import PubMeth
from twisted_mat.twisted_bi_gra import *


def get_fermi_vel_m0_r(m0_r_pair):
    tb_tbg = TightTbgInst(m0_r_pair[0], m0_r_pair[1])
    tmp_vel = tb_tbg.get_fermi_vel()
    return tmp_vel


def multi_get_fermi_vel(all_m0_r_pairs):
    parts_list = PubMeth.divide_list(all_m0_r_pairs)
    all_fermi_vel = PubMeth.multi_proc_func(get_fermi_vel_m0_r, parts_list, [])
    return all_fermi_vel
