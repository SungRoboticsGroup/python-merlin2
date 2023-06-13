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
import matplotlib.pyplot as plt

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
        analy_input_opt["rot_spr_fold"] = lambda he, h0, kf, l0, incl_espr: enhanced_linear(he, h0, kf, l0, 15, 345, incl_espr)
        analy_input_opt["load_type"] = "Force"
        analy_input_opt["initial_load_factor"] = 0.00001
        analy_input_opt["max_incr"] = 100
        analy_input_opt["stop_criterion"] = lambda node, u, icrm: np.abs(u[5*3 - 1]) > 12

    truss, angles, analy_input_opt = prepare_data(node, panel, supp, load, analy_input_opt) # type: ignore
    truss["u_0"] = np.zeros((3 * np.size(truss["node"], 0), 1))
    u_his, f_his = path_analysis(truss, angles, analy_input_opt, False)
    u_his = np.real(u_his)
    f_his = np.real(f_his)

    stat = post_process(u_his, truss, angles)
    instdof = np.array([5, -3])
    interv = 1
    endicrm = np.size(u_his, 1)
    visual_fold(u_his[:, : endicrm : interv], truss, angles, f_his[: endicrm : interv], instdof)
    f1 = plt.figure()
    ax1 = f1.add_subplot()
    dsp = np.sign(instdof[1]) * u_his[[instdof[0] * 3 - (3 - np.abs(instdof[1])) - 1]]

    ax1.plot(dsp.T, f_his, "-k")
    ax1.set_xlabel("Displacement", fontsize=14)
    ax1.set_ylabel("Force", fontsize=14)
    f1.tight_layout()
    plt.show(block=False)

    f2 = plt.figure()
    ax2 = f2.add_subplot()

    ax2.plot(dsp.flatten(), stat["pe"], "r-", linewidth = 2)
    ax2.set_xlabel("Displacement", fontsize=14)
    ax2.set_ylabel("Stored Energy", fontsize=14)
    f2.tight_layout()

    plt.show(block=False)

    f3 = plt.figure()
    ax3 = f3.add_subplot(projection="3d")
    plot_ori(ax3, truss["node"],angles["panel"],truss["trigl"].astype(int),fold_edge_style='-',edge_shade=0.3,panel_color=None)

    nodew = truss["node"].copy()
    ux = u_his[:, -1]
    nodew[:, 0] += ux[::3]
    nodew[:, 1] += ux[1::3]
    nodew[:, 2] += ux[2::3]
    plot_ori(ax3, nodew,angles["panel"],truss["trigl"].astype(int),panel_color='g')
    ax3.axis("off")
    plt.show(block=True)


test_single_plane(False)