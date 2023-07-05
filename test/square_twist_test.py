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
from src.util.load_file import load_dxf
import matplotlib.pyplot as plt
import scipy.sparse as sp
from src.util.save_data import save_object, load_object

def square_twist_test(do_n4b5=False):
    node, panel = load_dxf("sq2.dxf", True)
    m = np.size(node, 0)
    supp = np.array([[5, 0, 1, 0], [8, 0, 1, 0], [4, 0, 1, 1], [14, 0, 1, 1], [22, 0, 1, 0]])
    load = np.array([[22, 0, 0, 100], [5, 0, 0, 100], [8, 0, 0, 100]], dtype=float)


    analy_input_opt : AnalyInputOpt = {}
    analy_input_opt["model_type"] = "N5B8"
    analy_input_opt["mater_calib"] = "auto"
    analy_input_opt["mod_elastic"] = 1e3
    analy_input_opt["poisson"] = 0.3
    analy_input_opt["thickness"] = 0.25
    analy_input_opt["l_scale_factor"] = 2
    analy_input_opt["load_type"] = "displacement"
    analy_input_opt["disp_step"] = 200
    
    
    truss, angles, analy_input_opt = prepare_data(node, panel, supp, load, analy_input_opt)
    truss["u_0"] = np.zeros((3 * np.size(truss["node"], 0), 1))
    u_his, f_his = path_analysis(truss, angles, analy_input_opt, True)
    u_his = np.real(u_his)
    f_his = np.real(f_his)

    save_object("sq2", (truss, angles, analy_input_opt, u_his, f_his))
    truss, angles, analy_input_opt, u_his, f_his = load_object("sq2")
    stat = post_process(u_his, truss, angles)


    instdof = np.array([0, 2])
    interv = 1
    endicrm = np.size(u_his, 1)

    v_intensity_data_inten = np.zeros((np.size(truss["node"], 0), np.size(u_his, 1)))
    intensity_data_m = stat["bar"]["sx"] * truss["a"]
    for k in range(np.size(u_his, 1)):
        intensity_data_inten_k = np.zeros((np.size(truss["node"], 0), np.size(truss["node"], 0)))
        intensity_data_inten_k[truss["bars"][:, 0], truss["bars"][:, 1]] = np.abs(intensity_data_m[:, k])
        v_intensity_data_inten[:, k] = np.sum(intensity_data_inten_k + intensity_data_inten_k.T, 1)
    pass
    f = plt.figure()
    ax = f.add_subplot(projection="3d")
    nodew = truss["node"].copy()
    ux = u_his[:, -1]
    nodew[:, 0] += ux[::3]
    nodew[:, 1] += ux[1::3]
    nodew[:, 2] += ux[2::3]
    plot_ori(ax, nodew,angles["panel"],truss["trigl"].astype(int),panel_color='g')
    visual_fold(u_his[:, :endicrm:interv], truss, angles, f_his[:endicrm:interv], instdof, intensity_map="vertex", intensity_data = v_intensity_data_inten, record_type="imggif", filename="urmom")
    visual_fold(u_his[:, :endicrm:10], truss, angles, np.array([]), np.array([]), intensity_map = "edge", intensity_data = stat["bar"]["sx"][:, :endicrm:10], show_initial=False)
    visual_fold(u_his[:, :endicrm: interv], truss, angles, np.array([]), np.array([]))
    plt.show()

square_twist_test(False)