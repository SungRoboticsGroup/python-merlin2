from numpy.typing import NDArray
import numpy as np
from typing import Union, Tuple, List
from src.prepare_data import _nan
from mpl_toolkits.mplot3d import axes3d, art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

cache = {}

def _get_panels(panel):
    if type(panel[0]) == list:
        len_func = np.vectorize(len)
        panelsize = len_func(panel)
        panels = np.empty((np.size(panel, 0), np.max(panelsize)), dtype=int)
        for i in range(np.size(panel, 0)):
            panels[i, : panelsize[i]] = panel[i]
            panels[i, panelsize[i] : ] = panel[i][0]
        return panels
    else:
        return panel.astype(int)
    
def _interpolate_face_colors(big_tri, small_tri, colors_big_tri):
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

def _plot_3d(ax, prev = None, shade = False, **kwargs):
    """Either creates a new Poly3DCollection or updates an existing one, which is much faster. The figure must still be flushed after doing this."""
    if prev is None:
        p = Poly3DCollection(shade=shade, **kwargs)
        ax.add_collection3d(p)
        return p
    else:
        prev.set(**kwargs)
        return prev
    
def _plot_label(ax, prev = None, **kwargs):
    """Either creates a new text element or updates an existing one, which is much faster. The figure must still be flushed after doing this."""
    if prev is None:
        l = ax.text(**kwargs)
        return l
    else:
        prev.set(**kwargs)
        return prev
    
def _plot_plot(ax, prev = None, **kwargs) -> List[art3d.Line3D] :
    """Either creates a new text element or updates an existing one, which is much faster. The figure must still be flushed after doing this."""
    if prev is None:
        l = ax.plot(**kwargs)
        return l
    else:
        # This may cause future errors as line styles cannot be updated
        prev.set_data_3d(kwargs["xs"], kwargs["ys"], kwargs["zs"])
        return prev
    
def _get_prev(id, elem) -> List[art3d.Line3D] | None:
    if id is None:
        return None
    p = cache.get(id)
    if p == None:
        return None
    e = p.get(elem)
    return e

def _save_prev(id, elem, val):
    if id is not None:
        if cache.get(id) == None:
            cache[id] = {elem:val}
        else:
            cache[id][elem] = val

def clear_cache():
    """Clears the cached animation values. This is automatically done in visual_fold, and shouldn't need to be accessed by the user.
    """
    cache = {}

def plot_ori(
        ax : axes3d.Axes3D.__class__, 
        node : NDArray, 
        panel : NDArray, 
        trigl : NDArray | None, 
        fold_edge_style : str = "-",
        edge_shade : float = 1,
        panel_color : Union[None, str] = "g",
        bend_edge_style : str | None = None,
        show_number : bool = False,
        face_vertex_color : NDArray[np.float64] = np.array([]),
        edge_color : NDArray[np.float64] = np.array([]),
        bars : NDArray[np.float64] = np.array([]),
        num_bend_hinge : int = 0,
        face_shade : bool = False,
        id : str | None = None
    ) -> None :
    """This function generates origami renderings with various options. Among the inputs, Node and Panel are necessary. The following options are available:

    Parameters
    ----------
    ax : axes3d.Axes3D.__class__
        The 3d axes that the simulation should be rendered on. 
    node : NDArray
        The Nnode x 3 list of nodes
    panel : NDArray
        The Npanel x 3 list of panels
    trigl : NDArray | None
        The list of triangulated panels or None. If trigl is provided, the function will draw the triangulated origami model, which should be used for plotting deformed origami structures with bent panels.
    fold_edge_style : str, optional
        Matplotlib line style for the folding hinges, by default "-"
    edge_shade : float, optional
        A scalar between 0 and 1 that specifies the greyscale of edges in the plot. Use value 1 for black (default) and 0 for transparent, by default 1
    panel_color : Union[None, str], optional
        Specify a uniform colour for panels in any acceptable matplotlib color format, by default "g"
    bend_edge_style : str | None, optional
        Line style for the bending hinges. The default style is none, so that the origami plot does not display bending hinges. This property is only used when Trigl is provided, by default None
    show_number : bool, optional
        The plot_ori function shows the indices of nodes in the origami plot if this property is set to on. This is very helpful for specifying boundary conditions (i.e. Supp and Load). When this option is True, the panels are plotted without colour (transparent), by default False
    face_vertex_color : NDArray[np.float64], optional
        Face and vertex colours, specified as one colour per face, or one colour per vertex for interpolated face colour. For one colour per face, use an Nface × 3 array of RGB triplets, where Nf ace is the number of origami panels (panel) when trigl is not provided, or the number of triangles when trigl is provided. For interpolated face colour based on vertex values, provide an N'node ×3 array of RGB triplets, by default np.array([])
    edge_color : NDArray[np.float64], optional
        Colours for all bars in the bar-and-hinge model, specified as one colour per bar, including bending hinges, folding hinges, and boundary edges. Use an Nbar ×3 array of RGB triplets. To enable this property, connectivity of bar elements and Nbend needs to be specified. By default np.array([])
    bars : NDArray[np.float64], optional
        Pass in truss.bars if bars should be explicitly drawn, by default np.array([])
    num_bend_hinge : int, optional
        If bars is specified, the value of bending hinges should be specified here, by default 0
    face_shade : bool, optional
        Whether to shade the faces, by default False
    id : str | None, optional
        This field is used for animating with plot_ori. When an id is specified the objects that are drawn to ax are cached and their data is updated, rather than clearing ax and instantiating new surfaces every tick. This is significantly faster for animation, and care should be taken to ensure that unique ids are chosen OR clear_cache() is called to remove cached animation data. ax should not be cleared if animating. If None, then the drawing information will not be cached. By default None

    Raises
    ------
    ValueError
        If edge coloring is used without passing in triangulation info
    """

    panels = _get_panels(panel)
    coords_tri = node[trigl]
    coords_pan = node[panels]

    axislim = np.hstack([np.min(node, 0), np.max(node, 0)])
    ax.axes.set_xlim3d(left=axislim[0], right=axislim[3]) 
    ax.axes.set_ylim3d(bottom=axislim[1], top=axislim[4]) 
    ax.axes.set_zlim3d(bottom=axislim[2], top=axislim[5])
    ax.set_aspect("equal")


    if show_number:
        panel_color = None
        bend_edge_style = ":"
    if type(face_vertex_color) == np.ndarray and np.size(face_vertex_color) != 0:
        if np.size(face_vertex_color, 0) == np.size(node, 0):
            # this is intended to replace interpolation
            face_vertex_color = np.average(face_vertex_color[panels], 1)
        panel_color = "flat"
    if edge_color.size == 0:
        if trigl is not None and trigl.size != 0:

            p1 = p2 = None

            if panel_color in ["flat", "interp"]:
                p1 = _plot_3d(ax, _get_prev(id, "p1"), verts = coords_pan, facecolor=face_vertex_color, linestyles=fold_edge_style, linewidths=1, edgecolors=np.full((1, 3), 1 - edge_shade))

                # pt = ax.plot_trisurf(node[:, 0], node[:, 1], node[:, 2], color=face_vertex_color, edgecolor = np.full((1, 3), 1 - edge_shade), linestyle = bend_edge_style)
            # elif panel_color == "interp":
                # # from https://stackoverflow.com/questions/19836199/interpolating-a-3d-surface-known-by-its-corner-nodes-and-coloring-it-with-a-colo
                # pt = plot_trisurf_panels(ax, coords_pan[:, 0], coords_pan[:, 1], coords_pan[:, 2], facecolors="g") #TODO: this is gonna be annoying
            else:
                kwargs = {
                    "edgecolors": np.full((1, 3), 1 - edge_shade),
                    "facecolors": panel_color if panel_color is not None else (0,0,0,0)
                }
                if bend_edge_style is not None:
                    kwargs["linestyle"] = bend_edge_style
                else:
                    kwargs["edgecolors"] = (0,0,0,0)
                p1 = _plot_3d(ax, _get_prev(id, "p1"), verts = coords_tri, shade=face_shade, **kwargs)

                # TODO: refactor all facecolors to color
            p1.set_sort_zpos(10) # type: ignore
            _save_prev(id, "p1", p1)


        if trigl is not None and trigl.size != 0:
            p2 = _plot_3d(ax, _get_prev(id, "p2"), verts = coords_pan, facecolor=(0,0,0,0), linestyles=fold_edge_style, linewidths=1, edgecolors=np.full((1, 3), 1 - edge_shade), zorder = 0)
        else:
            if panel_color == "flat":
                p2 = _plot_3d(ax, _get_prev(id, "p2"), verts=coords_pan, facecolor=face_vertex_color, linestyles=fold_edge_style, linewidths=1, edgecolors=np.full((1, 3), 1 - edge_shade))
            else:
                p2 = _plot_3d(ax, _get_prev(id, "p2"), verts = coords_pan, facecolor=panel_color, linestyles=fold_edge_style, linewidths=1, edgecolors=np.full((1, 3), 1 - edge_shade))
        
        p2.set_sort_zpos(0)
        _save_prev(id, "p2", p2)

        node_labels = _get_prev(id, "no_edge_labels")
        if node_labels is None: node_labels = [None for _ in range(np.size(node))]
        if show_number:
            for i in range(np.size(node, 0)):
                node_labels[i] = _plot_label(ax, node_labels[i], x = node[i, 0] + 0.1, y = node[i, 1] - 0.1, z = node[i, 2], text = str(i), fontsize = 14)

        _save_prev(id, "no_edge_labels", node_labels)
    
    else:
        if trigl is not None and trigl.size == 0:
            raise ValueError("Edge Coloring mode requires triangulation info")
        p3 = _plot_3d(ax, _get_prev(id, "p3"), verts = coords_pan, facecolor=tuple(0.85 for i in range(3)) + (0.8,), edgecolors=(0,0,0,0))
        _save_prev(id, "p3", p3)

        bend_lines = _get_prev(id, "bend_lines")
        if bend_lines is None: 
            bend_lines = [None for _ in range(num_bend_hinge)]

        for i in range(num_bend_hinge):
            xyz = np.hstack((node[bars[i, 0]], node[bars[i, 1]]))
            bend_lines[i] = _plot_plot(ax, bend_lines[i], xs = xyz[[0,3]].T, ys = xyz[[1, 4]].T, zs = xyz[[2,5]].T, linestyle = ":", linewidth=1.5, color=edge_color[i], zorder = 5) # type: ignore
            if type(bend_lines[i]) == list:
                assert bend_lines[i] is not None
                bend_lines[i] = bend_lines[i][0] # type: ignore
        _save_prev(id, "bend_lines", bend_lines)

        bar_lines : List[art3d.Line3D] | List[None] | None = _get_prev(id, "bar_lines")
        if bar_lines is None: bar_lines = [None for _ in range(np.size(bars, 0))]
        for j in range(num_bend_hinge, np.size(bars, 0)):
            xyz = np.hstack((node[bars[j, 0]], node[bars[j, 1]]))
            bar_lines[j - num_bend_hinge] = _plot_plot(ax, bar_lines[j - num_bend_hinge], xs = xyz[[0,3]].T, ys = xyz[[1, 4]].T, zs = xyz[[2,5]].T, linestyle = "-", linewidth=2, color=edge_color[j]) # type: ignore
            if type(bar_lines[j - num_bend_hinge]) == list:
                bar_lines[j - num_bend_hinge] = bar_lines[j - num_bend_hinge][0] # type: ignore
        _save_prev(id, "bar_lines", bar_lines)
    
    if show_number:
        edge_node_labels = _get_prev(id, "edge_node_labels")
        if edge_node_labels is None: edge_node_labels = [None for _ in range(np.size(node, 0))]
        for i in range(np.size(node, 0)):
            edge_node_labels[i] = _plot_label(ax, edge_node_labels[i], x = node[i, 0] + 0.1, y = node[i, 1] - 0.1, z = node[i, 2], s = str(i), fontsize = 14)