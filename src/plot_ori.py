from numpy.typing import NDArray
import numpy as np
from typing import Union
from src.prepare_data import _nan
from matplotlib import tri
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def interpolate_face_colors(big_tri, small_tri, colors_big_tri):
    # find the center point of each small triangle, and find the closest large triangle point. 
    ctrs = np.average(small_tri, 1).reshape(-1, 3)
    max_bigtri = np.max(big_tri, 1).reshape(-1, 3)
    min_bigtri = np.min(big_tri, 1).reshape(-1, 3)
    colors = _nan(np.size(ctrs, 0), 3)
    for i in range(np.size(ctrs, 0)):
        cnt = np.all(np.bitwise_and(max_bigtri >= ctrs[i], min_bigtri <= ctrs[i]), axis=1)
        tri_ind = np.where(cnt)[0]
        assert np.size(tri_ind) == 1
        tri = big_tri[tri_ind[0]]
        dists = np.linalg.norm(tri - ctrs[i], axis=1)
        colors[i] = np.average(colors_big_tri[tri], axis=0, weights=dists)
    
    return colors

def _plot_trisurf_panels(ax, x, y, z, subdiv = 3, **kwargs):
    tri_small = tri.Triangulation(x, y)
    refiner = tri.UniformTriRefiner(tri_small)
    interpolator = tri.LinearTriInterpolator(tri_small, z)
    new, new_z = refiner.refine_field(z, interpolator, subdiv=subdiv)
    return ax.plot_trisurf(new.x, new.y, new_z, triangles=new.triangles, **kwargs) #TODO: if facecolors are included, have to expand
   
def _get_panels(panel):
    if type(panel[0]) == list:
        len_func = np.vectorize(len)
        panelsize = len_func(panel)
        panels = _nan(np.size(panel, 0), np.max(panelsize))
        for i in range(np.size(panel, 0)):
            panels[i, : panelsize[i]] = panel[i]
        return _get_panels(panels)
    else:
        return panel.astype(int)
    
def _patch(ax, x, y, z, facecolor=(0,0,0,0), edgecolor='k'):
    pc = Poly3DCollection([list(zip(x,y,z))])       # Create PolyCollection from coords
    pc.set_facecolor(facecolor)                             # Set facecolor to mapped value
    pc.set_edgecolor(edgecolor)                           # Set edgecolor to black
    ax.add_collection3d(pc)                         # Add PolyCollection to axes
    return pc


def plot_ori(
        ax : axes3d.Axes3D.__class__, 
        node : NDArray, 
        panel : NDArray, 
        trigl : NDArray, 
        fold_edge_style : str = "-",
        edge_shade : float = 1,
        panel_color : Union[None, str] = "g",
        bend_edge_style : str | None = None,
        show_number : bool = False,
        face_vertex_color : NDArray[np.float64] = np.array([]),
        edge_color : NDArray[np.float64] = np.array([]),
        bars : NDArray[np.float64] = np.array([]),
        num_bend_hinge : int = 0
    ):

    

    panels = _get_panels(panel)
    coords_tri = node[trigl]
    coords_pan = node[panels]

    if show_number:
        panel_color = None
        bend_edge_style = ":"
    if type(face_vertex_color) == np.ndarray and np.size(face_vertex_color) != 0:
        if np.size(face_vertex_color, 0) == np.size(node, 0):
            # this is intended to replace interpolation
            face_vertex_color = np.average(face_vertex_color[panel], 1)
        panel_color = "flat"
    if edge_color.size == 0:
        if trigl.size != 0:
            if panel_color in ["flat", "interp"]:
                for i in range(np.size(coords_pan, 0)):
                    cpi = coords_pan[i]
                    cpi = np.vstack((cpi, cpi[0]))
                    pn = ax.plot_wireframe(cpi[:, [0]], cpi[:, [1]], cpi[:, [2]], facecolor=face_vertex_color[i], linestyles=fold_edge_style, linewidths=1, edgecolors=np.full((1, 3), 1 - edge_shade))
                pass
                # pt = ax.plot_trisurf(node[:, 0], node[:, 1], node[:, 2], color=face_vertex_color, edgecolor = np.full((1, 3), 1 - edge_shade), linestyle = bend_edge_style)
            # elif panel_color == "interp":
                # # from https://stackoverflow.com/questions/19836199/interpolating-a-3d-surface-known-by-its-corner-nodes-and-coloring-it-with-a-colo
                # pt = plot_trisurf_panels(ax, coords_pan[:, 0], coords_pan[:, 1], coords_pan[:, 2], facecolors="g") #TODO: this is gonna be annoying
            else:
                kwargs = {
                    "edgecolors": np.full((1, 3), 1 - edge_shade),
                    "color": panel_color if panel_color is not None else (0,0,0,0)
                }
                if bend_edge_style is not None:
                    kwargs["linestyle"] = bend_edge_style
                else:
                    kwargs["edgecolor"] = (0,0,0,0)
                pt = ax.plot_trisurf(node[:, 0], node[:, 1], node[:, 2], **kwargs)
                # TODO: refactor all facecolors to color

        if trigl.size != 0:
            for i in range(np.size(coords_pan, 0)):
                cpi = coords_pan[i]
                cpi = np.vstack((cpi, cpi[0]))
                pn = ax.plot_wireframe(cpi[:, [0]], cpi[:, [1]], cpi[:, [2]], facecolor=(0,0,0,0), linestyles=fold_edge_style, linewidths=1, edgecolors=np.full((1, 3), 1 - edge_shade))
        else:
            if panel_color == "flat":
                pn = _plot_trisurf_panels(ax, coords_pan[:, 0], coords_pan[:, 1], coords_pan[:, 2], linestyles=fold_edge_style, linewidths=1, edgecolors=np.full((1, 3), 1 - edge_shade), color=face_vertex_color)
            else:
                pn = _plot_trisurf_panels(ax, coords_pan[:, 0], coords_pan[:, 1], coords_pan[:, 2], linestyles=fold_edge_style, linewidths=1, edgecolors=np.full((1, 3), 1 - edge_shade), color=panel_color)

        if show_number:
            for i in range(np.size(node)):
                ax.text(node[i, 0] + 0.1, node[i, 1] - 0.1, node[i, 2], str(i), fontsize = 14)
    
    else:
        if trigl.size == 0:
            raise ValueError("Edge Coloring mode requires triangulation info")
        for i in range(np.size(coords_tri, 0)):
            cpi = coords_tri[i]
            cpi = np.vstack((cpi, cpi[0]))
            pn = ax.plot_wireframe(cpi[:, [0]], cpi[:, [1]], cpi[:, [2]], facecolors=tuple(0.85 for i in range(3)) + (0.8,), edgecolors = (0,0,0,0), zorder=0)
        for i in range(num_bend_hinge):
            xyz = np.hstack((node[bars[i, 0]], node[bars[i, 1]]))
            ax.plot(xyz[[0,3]].T, xyz[[1, 4]].T, xyz[[2,5]].T, ":", linewidth=1.5, color=edge_color[i],zorder = 5)

        for j in range(num_bend_hinge, np.size(bars, 0)):
            xyz = np.hstack((node[bars[j, 0]], node[bars[j, 1]]))
            ax.plot(xyz[[0,3]].T, xyz[[1, 4]].T, xyz[[2,5]].T, "-", linewidth=2, color=edge_color[j])
    if show_number:
        for i in range(np.size(node, 0)):
            ax.text(node[i, 0] + 0.1, node[i, 1] - 0.1, node[i, 2], str(i), fontsize = 14)