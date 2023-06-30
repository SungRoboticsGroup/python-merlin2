import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import axes
import matplotlib.animation as anim
from mpl_toolkits.mplot3d.axes3d import Axes3D
from src.plot_ori import plot_ori, _get_panels, clear_cache
from time import sleep
from src.prepare_data import _nan

# mpl.use("Qt5agg")


def visual_fold(
        u_his, 
        truss, 
        angles, 
        lf_his, 
        instdof, 
        record_type=None, 
        show_initial=True, 
        filename="", 
        pause_time = 0.0001, 
        axislim = None,
        intensity_map = "off",
        intensity_data = np.array([]),
        view_angle = (35.0, 30.0)):
    node = truss["node"]
    trigl = truss["trigl"].astype(int)
    panel = angles["panel"]
    u_his = np.hstack((truss["u_0"], u_his))
    col_col = None
    v_intensity_data_inten_index = None
    e_intensity_data_inten_index = None
    f_intensity_data_inten_index = None

    clear_cache()

    if intensity_map == "vertex":
        if np.size(intensity_data) > 0 and np.min(intensity_data) >= 0:
            ng = 110
            col_col = mpl.colormaps["viridis"](np.linspace(-1, 1, ng)) # type: ignore
            v_intensity_data_inten = np.hstack((intensity_data[:, [0]] * 0, intensity_data))/np.max(np.abs(intensity_data))
        else:
            cmp = mpl.colormaps["hsv"](np.linspace(-1, 1, 120)) # type: ignore
            cmpr = np.zeros((10, 3))
            cmpb = cmpr.copy()
            cmpr[:, 0] = np.arange(0.5, 1, 0.05)
            cmpb[:, 2] = np.arange(1, 0.5, 0.05)
            col_col = np.vstack((cmpr, cmp[: 21], cmp[21:61:3], cmp[61:81], cmpb))
            ng = np.size(col_col, 0)
            v_intensity_data_inten = 0.5 * (np.clip(np.hstack((0*intensity_data[:, 0], intensity_data)) / np.max(np.abs(intensity_data)), -1, 1) + 1)
        v_intensity_data_inten_index = (np.ceil((ng - 10) * v_intensity_data_inten) + 9).astype(int)
    elif intensity_map == "edge":
        if np.min(intensity_data) >= 0:
            ng = 110
            col_col = mpl.colormaps["viridis"](np.linspace(-1, 1, ng)) # type: ignore
            e_intensity_data_inten = np.hstack((intensity_data[:, 0] * 0, intensity_data))/np.max(np.abs(intensity_data))
        else:
            cmp = mpl.colormaps["hsv"](np.linspace(-1, 1, 120))[:, :3] # type: ignore
            cmpr = np.zeros((10, 3))
            cmpb = cmpr.copy()
            cmpr[:, 0] = np.arange(0.5, 1, 0.05)
            cmpb[:, 2] = np.arange(0.95, 0.45, -0.05)
            col_col = np.vstack((cmpr, cmp[: 21], cmp[21:61:3], cmp[61:81], cmpb))
            ng = np.size(col_col, 0)
            e_intensity_data_inten = 0.5 * (np.clip(np.hstack((0*intensity_data[:, [0]], intensity_data)) / np.max(np.abs(intensity_data)), -1, 1) + 1)
        e_intensity_data_inten_index = (np.ceil((ng - 10) * e_intensity_data_inten) + 9).astype(int)
    elif intensity_map == "face":
        if np.min(intensity_data) >= 0:
            ng = 110
            col_col = mpl.colormaps["viridis"](np.linspace(-1, 1, ng)) # type: ignore
            f_intensity_data_inten = np.hstack((intensity_data[:, 0] * 0, intensity_data))/np.max(np.abs(intensity_data))
        else:
            cmp = mpl.colormaps["hsv"](np.linspace(-1, 1, 120)) # type: ignore
            cmpr = np.zeros((10, 3))
            cmpb = cmpr.copy()
            cmpr[:, 0] = np.arange(0.5, 1, 0.05)
            cmpb[:, 2] = np.arange(1, 0.5, 0.05)
            col_col = np.vstack((cmpr, cmp[: 21], cmp[21:61:3], cmp[61:81], cmpb))
            ng = np.size(col_col, 0)
            f_intensity_data_inten = 0.5 * (np.clip(np.hstack((0*intensity_data[:, 0], intensity_data)) / np.max(np.abs(intensity_data)), -1, 1) + 1)
        f_intensity_data_inten_index = (np.ceil((ng - 10) * f_intensity_data_inten) + 9).astype(int)

    use_lf = lf_his is not None and lf_his.size > 0
    if use_lf:
        lf_his = np.vstack((0 * lf_his[[0], :], lf_his))
        if np.size(lf_his, 1) > 1:
            lf_his = np.sum(lf_his, 1, keepdims=True)
    writer = None
    if record_type == "video":
        writer = anim.FFMpegWriter(fps = 24)
    elif record_type == "imggif":
        writer = anim.ImageMagickWriter(fps = 24) #TODO this may be wrong
    else:
        print("Not recording")
    
    f1 = plt.figure()
    f1.set_facecolor("w")

    if writer is not None:
        writer.setup(f1, filename + (".gif" if record_type == "imggif" else ".mp4"))

    if axislim is None:
        axislim = np.hstack([np.min(node, 0), np.max(node, 0)])

    ax : Axes3D.__class__ = f1.add_subplot(projection="3d")
    f1.show()

    ax.axes.set_xlim3d(left=axislim[0], right=axislim[3]) 
    ax.axes.set_ylim3d(bottom=axislim[1], top=axislim[4]) 
    ax.axes.set_zlim3d(bottom=axislim[2], top=axislim[5])
    ax.view_init(*view_angle)
    ax.set_aspect("equal")
    ax.axis("off")
    marker = None
    
    #TODO: figure this out , position=100 + np.array([0, 0, 720, 500])
    for i in range(np.size(u_his, 1)):
        u = u_his[:, i]
        nodew = node.copy()
        nodew[:, 0] = node[:, 0] + u[::3]
        nodew[:, 1] = node[:, 1] + u[1::3]
        nodew[:, 2] = node[:, 2] + u[2::3]

             
        if show_initial:
            plot_ori(ax, node, panel, trigl, fold_edge_style="-", edge_shade=0.3, panel_color=None, id = "init")
        
        if intensity_map == "vertex" and col_col is not None and v_intensity_data_inten_index is not None:
            plot_ori(ax, nodew, panel, trigl, face_vertex_color=col_col[v_intensity_data_inten_index[:, i]], id = "d")
        elif intensity_map == "edge" and col_col is not None and e_intensity_data_inten_index is not None:
            plot_ori(ax, nodew, panel, trigl, edge_color=col_col[e_intensity_data_inten_index[:, i]], bars=truss["bars"], num_bend_hinge=np.size(angles["bend"], 0), id = "d")
        elif intensity_map == "face" and col_col is not None and f_intensity_data_inten_index is not None:
            plot_ori(ax, nodew, panel, trigl, face_vertex_color=col_col[f_intensity_data_inten_index[:, i]], id = "d")
        else:
            plot_ori(ax, nodew, panel, trigl, id = "d")
    
        if use_lf:
            if marker is None:
                marker = ax.plot(nodew[instdof[0], 0], nodew[instdof[0], 1], nodew[instdof[0], 2], "rv", markeredgewidth=2, markersize=10)
                if type(marker) == list:
                    marker = marker[0]
            else:
                marker.set_data_3d([nodew[instdof[0], 0]], [nodew[instdof[0], 1]], [nodew[instdof[0], 2]])

        # sleep(pause_time)
        #TODO: add pause time back but make it based on a fps rather than pause per frame
        f1.canvas.draw() # draw
        f1.canvas.flush_events() # deal with resize

        if writer is not None:
            writer.grab_frame()
    if writer is not None:
        writer.finish()

    if use_lf:
        filename += "_dispvslambda"

        writer = None
        if record_type == "video":
            writer = anim.FFMpegFileWriter(fps = 24)
        elif record_type == "imggif":
            writer = anim.ImageMagickFileWriter(fps = 24) #TODO this may be wrong
        
        f2 = plt.figure()

        f2.set_facecolor("w")
        dsp = np.sign(instdof[1]) * u_his[[instdof[0] * 3 - (3 - np.abs(instdof[1])) - 1]]

        if writer is not None:
            writer.setup(f2, filename + (".gif" if record_type == "imggif" else ".mp4"))

        ax2 : axes.Axes = f2.add_subplot()
        f2.show()
        #TODO: figure this out , position=100 + np.array([0, 0, 720, 500])
        for i in range(np.size(lf_his, 1)):
            # ax.view_init(*view_angle)

            ax2.plot(dsp[:i], lf_his[:i], 'b-', linewidth=2)
            ax2.plot(dsp[i], lf_his.T[i], "ro", markeredgewidth=2)
            ax2.set_xlabel("Dispacement", fontsize=16)
            ax2.set_ylabel("Load Factor", fontsize=16)
            f2.canvas.draw() # draw
            f2.canvas.flush_events() # deal with resize

            sleep(pause_time)

            if writer is not None:
                writer.grab_frame()
        if writer is not None:
            writer.finish()
    
    pass