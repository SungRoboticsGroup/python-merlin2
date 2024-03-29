import numpy as np
import sys
import scipy.sparse as sp
import matplotlib.pyplot as plt

# hacky fix for import issues bc of package structuring, taken from https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve() # type: ignore
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass

from src import *
from src.util import *
# from prepare_data import prepare_data, AnalyInputOpt
# from load_obj import load_obj
# from ogden import ogden
# from enhanced_linear import enhanced_linear

do_n4b5 = False

node, panel = load_obj("example/GMiura_FreeformOri.obj")
m = np.size(node, 0)
supp = np.array([[7, 1, 1, 1], [1, 0, 1, 1], [58, 1, 0, 1]])
indp = 58
ff = -30 * np.ones((np.size(indp),), dtype = int)
load = np.hstack([indp, np.zeros((np.size(indp),)), ff, np.zeros((np.size(indp),))]).reshape((1, -1)).astype(int)
if not do_n4b5:
    analy_input_opt : AnalyInputOpt = {
        "model_type": "N5B8",
        "mater_calib": "auto",
        "mod_elastic": 1e3,
        "poisson": 0.3,
        "thickness": 0.25,
        "l_scale_factor": 2,
        "load_type":"displacement",
        "disp_step":200
    }
else:
    analy_input_opt : AnalyInputOpt = {
        "model_type": "N4B5",
        "mater_calib": "manual",
        "bar_cm": lambda ex, incl: ogden(ex, 1e4, incl),
        "a_bar": 2e-1,
        "K_b": 0.3,
        "K_f": 0.03,
        "rot_spr_bend": lambda he, h0, kb, l0, incl: enhanced_linear(he, h0, kb, l0, 30, 330, incl),
        "rot_spr_fold": lambda he, h0, kb, l0, incl: enhanced_linear(he, h0, kb, l0, 30, 330, incl),
        "load_type":"displacement",
        "disp_step":200
    }


truss, angles, analy_input_opt = prepare_data(node, panel, supp, load, analy_input_opt)
truss["u_0"] = np.zeros((3 * np.size(truss["node"], 0), 1))
u_his, f_his = path_analysis(truss, angles, analy_input_opt, True)
u_his = np.real(u_his)
f_his = np.real(f_his)

save_object("miura_test", (truss, angles, analy_input_opt, u_his, f_his))
# truss, angles, analy_input_opt, u_his, f_his = load_object("miura_test")

stat = post_process(u_his, truss, angles)

instdof = np.array([indp, 2])
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

plt.show(block = True)