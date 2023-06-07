from numpy.typing import NDArray
import numpy as np
from mpl_toolkits.mplot3d import art3d


def plot_ori(
        node : NDArray, 
        panel : NDArray, 
        trigl : NDArray, 
        fold_edge_style : str = "-",
        edge_shade : float = 1,
        panel_color : None | str = "g",
        bend_edge_style : str = "",
        show_number : bool = False,
        face_vertex_color : NDArray[np.float64] = np.array([]),
        edge_color : NDArray[np.float64] = np.array([]),
        bars : NDArray[np.float64] = np.array([]),
        num_bend_hinge : int = 0
    ):
        if show_number:
            panel_color = None
            bend_edge_style = ":"
        if np.size(face_vertex_color) != 0:
            if np.size(face_vertex_color, 0) == np.size(node, 0):
                panel_color = "interp"
            else:
                panel_color = "flat"
        if edge_color.size == 0:
            if trigl.size != 0:
                p = art3d.Poly3DCollection(node[trigl])
                p.set(facecolor=panel_color, )
