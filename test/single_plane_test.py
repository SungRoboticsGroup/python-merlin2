import numpy as np
from src.util.dicts import AnalyInputOpt
from src.util.ogden import ogden
from src.util.super_linear_bend import super_linear_bend
from src.util.enhanced_linear import enhanced_linear
from src.prepare_data import prepare_data
from src.path_analysis import path_analysis

def test_single_plane(do_n4b5=False):
    node = np.array([[0, 0, 0], [0, 10, 0], [12, 15, 0], [30, 10, 0], [30, 0, 0], [18, -5, 0]]) * 2
    panel = np.empty((1,), dtype=object)
    panel[0] = list(range(6)) # important to make sure panel is in the right format

    m = np.size(node, 0)
    supp = np.array([[0, 1, 1, 1], [1, 1, 0, 1], [3, 0, 0, 1]])
    load = np.array([[4, 0, 0, -1]])

    analy_input_opt : AnalyInputOpt = {}
    if not do_n4b5:
        analy_input_opt["model_type"] = "N5B8"
        analy_input_opt["mater_calib"] = "auto"
        analy_input_opt["mod_elastic"] = 5e3
        analy_input_opt["poisson"] = 0.35
        analy_input_opt["thickness"] = 0.127
        analy_input_opt["l_scale_factor"] = 3
        analy_input_opt["load_type"] = "force"
        analy_input_opt["initial_load_factor"] = 0.00001
        analy_input_opt["max_incr"] = 100 
        analy_input_opt["stop_criterion"] = lambda node, u, icrm: np.abs(u[5*3 - 1]) > 12
    else:
        analy_input_opt["model_type"] = "N4B5"
        analy_input_opt["mater_calib"] = "manual"
        analy_input_opt["bar_cm"] = lambda ex, incl: ogden(ex, 5e3, incl)
        analy_input_opt["a_bar"] = 2.4652
        analy_input_opt["K_b"] = 0.9726 * np.divide(np.power(np.array([38.4187, 38.4187, 41.7612])/0.127, 1/3), np.array([38.4187, 38.4187, 41.7612]))
        analy_input_opt["K_f"] = 1
        analy_input_opt["rot_spr_bend"] = super_linear_bend
        analy_input_opt["rot_spr_fold"] = lambda he, h0, kf, l0: enhanced_linear(he, h0, kf, l0, 15, 345)
        analy_input_opt["load_type"] = "Force"
        analy_input_opt["initial_load_factor"] = 0.00001
        analy_input_opt["max_incr"] = 100
        analy_input_opt["stop_criterion"] = lambda node, u, icrm: np.abs(u[5*3 - 1] > 12)

    truss, angles, analy_input_opt = prepare_data(node, panel, supp, load, analy_input_opt)
    truss["u_0"] = np.zeros((3 * np.size(truss["node"], 0), 1))
    u_his, f_his = path_analysis(truss, angles, analy_input_opt)
    u_his = np.real(u_his)
    f_his = np.real(f_his)
    pass

test_single_plane(False)