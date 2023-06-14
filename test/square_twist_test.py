import numpy as np
from src.util.dicts import AnalyInputOpt
from src.util.ogden import ogden
from src.util.super_linear_bend import super_linear_bend
from src.util.enhanced_linear import enhanced_linear
from src.prepare_data import prepare_data
from src.path_analysis import path_analysis
from src.post_process import post_process
from src.visual_fold import visual_fold
from src.plot_ori import plot_ori
from src.util.load_dxf import load_dxf
import matplotlib.pyplot as plt
import scipy.sparse as sp

def square_twist_test(do_n4b5=False):
    node, panel = load_dxf("sq2.dxf", True)
    m = np.size(node, 0)
    supp = np.array([[4, 1, 1, 1], [6, 1, 1, 1], [13, 1, 1, 1], [16, 1, 1, 1]])
    load = np.array([[0, 1, -1, 0.15], [23, -1, 1, 0.15]], dtype=float)


    analy_input_opt : AnalyInputOpt = {}
    analy_input_opt["model_type"] = "N5B8"
    analy_input_opt["mater_calib"] = "auto"
    analy_input_opt["mod_elastic"] = 5e3
    analy_input_opt["poisson"] = 0.35
    analy_input_opt["thickness"] = 0.127
    analy_input_opt["l_scale_factor"] = 3
    analy_input_opt["load_type"] = "force"
    analy_input_opt["initial_load_factor"] = 0.01
    analy_input_opt["max_incr"] = 300 
    analy_input_opt["stop_criterion"] = lambda node, u, icrm: icrm == 300
    
    
    truss, angles, analy_input_opt = prepare_data(node, panel, supp, load, analy_input_opt)
    truss["u_0"] = np.zeros((3 * np.size(truss["node"], 0), 1))
    u_his, f_his = path_analysis(truss, angles, analy_input_opt, True)
    u_his = np.real(u_his)
    f_his = np.real(f_his)

    stat = post_process(u_his, truss, angles)

    instdof = np.array([0, 2])
    interv = 1
    endicrm = np.size(u_his, 1)

    v_intensity_data_inten = np.zeros((np.size(truss["node"], 0), np.size(u_his, 1)))
    intensity_data_m = stat["bar"]["sx"] * truss["a"]
    for k in range(np.size(u_his, 1)):
        intensity_data_inten_k = sp.csr_array((np.abs(intensity_data_m[:, k]), (truss["bars"][:, 0], truss["bars"][:, 1])), tuple(np.size(truss["node"], 0) for i in range(2)))
        v_intensity_data_inten[:, k] = np.sum(intensity_data_inten_k + intensity_data_inten_k.T, 1)
    pass
    visual_fold(u_his[:, :endicrm:interv], truss, angles, f_his[:endicrm:interv], instdof, intensity_map="vertex", intensity_data = v_intensity_data_inten, record_type = "imggif", filename="miura_test")
    visual_fold(u_his[:, :endicrm:10], truss, angles, np.array([]), np.array([]), intensity_map = "edge", intensity_data = stat["bar"]["sx"][:, :endicrm:10], show_initial=False)
    visual_fold(u_his[:, :endicrm: interv], truss, angles, np.array([]), np.array([]))


square_twist_test(False)