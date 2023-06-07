import numpy as np
import matplotlib as mpl

def visual_fold(
        u_his, 
        truss, 
        angles, 
        lf_his, 
        instdof, 
        record_type=None, 
        show_initial="on", 
        filename=None, 
        pause_time = 0.0001, 
        axislim = np.array([-np.inf, np.inf, -np.inf, np.inf]),
        intensity_map = "off",
        intensity_data = np.array([]),
        view_angle = (35.0, 30.0)):
    node = truss["node"]
    trigl = truss["trigl"]
    panel = angles["panel"]
    u_his = np.hstack((truss["u_0"], truss["trigl"]))
    if intensity_map == "vertex":
        if np.size(intensity_data) > 0 and np.min(intensity_data, 0) >= 0:
            ng = 110
            # col_col = 
