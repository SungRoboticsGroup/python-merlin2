import numpy as np
import sys

# hacky fix for import issues bc of package structuring, taken from https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass

from src import *
# from prepare_data import prepare_data, AnalyInputOpt
# from load_obj import load_obj
# from ogden import ogden
# from enhanced_linear import enhanced_linear

do_n4b5 = False

node, panel = load_obj("example/GMiura_FreeformOri.obj")
m = np.size(node, 0)
supp = np.array([[8, 1, 1, 1], [2, 0, 1, 1], [59, 1, 0, 1]])
indp = 59
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
        "bar_cm": lambda ex: ogden(ex, 1e4),
        "a_bar": 2e-1,
        "k_b": 0.3,
        "k_f": 0.03,
        "rot_spr_bend": lambda he, h0, kb, l0: enhanced_linear(he, h0, kb, l0, 30, 330),
        "rot_spr_fold": lambda he, h0, kb, l0: enhanced_linear(he, h0, kb, l0, 30, 330),
        "load_type":"displacement",
        "disp_step":200
    }


truss, angles, analy_input_opt = prepare_data(node, panel, supp, load, analy_input_opt)
truss["u_0"] = np.zeros((3 * np.size(truss["node"], 0), 1))
u_his, f_his = path_analysis(truss, angles, analy_input_opt)
u_his = np.real(u_his)
f_his = np.real(f_his)

pass
