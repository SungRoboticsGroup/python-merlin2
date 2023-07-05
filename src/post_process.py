import numpy as np
import numpy.typing as npt
from src.util.dicts import Truss, Angles, Fold, Bend, Bar, Stat
from src.util.fold_ke import fold_ke

def post_process(u_his : npt.NDArray, truss : Truss, angles: Angles):
    """This function computes physical measures for the bar-andhinge model based on the deformation history stored in u_his, such as strains of bar elements and system energies. 

    Parameters
    ----------
    u_his : npt.NDArray
        The u_his output from path_analysis
    truss : Truss
        The truss output from prepare_data
    angles : Angles
        The angles output from prepare_data

    Returns
    -------
    stat : Stat
        All measures are stored in a dict. A complete list of outputs are shown below:
        \n1. bar : Bar - a dict containing nformation about bar elements at every increment: 
            \na. Green-Lagrange strain (bar.ex)
            \nb. 2nd PK stress (bar.sx)
            \nc. Stored energy of each bar element (bar.us_i)
            \nd. Total stored energy of bar elements (bar.us)
        \nAttributes bar.ex, bar.sx, and bar.us_i are of size Nbar ×Nicrm. Attribute bar.us is a 1×Nicrm array.
        \n2. fold : Fold - A dict containing information about folding rotational springs at every increment: 
            \na. Folding angle (fold.angle)
            \nb. Resistant moment (fold.rm)
            \nc. Stored energy of each folding hinge (fold.uf_i)
            \nd. Total stored energy of folding hinges (fold.uf).
        \nAttributes fold.angle, fold.r_m, and fold.uf_i are of size Nf old × Nicrm. Attribute fold.uf is a 1×Nicrm array.
        \n3. bend - Same as STAT.fold, but for bending rotational springs, which also has
        4 attributes:
            \na. bending angle (bend.angle)
            \nb. resistant moment (bend.rm)
            \nc. stored energy of each bending hinge (bend.ub_i)
            \nd. total stored energy (bend.ub).
        \n4. pe : npt.NDArray - Total potential energy of the origami structure, stored in a 1×Nicrm array
    """
    ex_bar = np.zeros((np.size(truss["bars"], 0), np.size(u_his, 1)))
    fd_angle = np.zeros((np.size(angles["fold"], 0), np.size(u_his, 1)))
    bd_angle = np.zeros((np.size(angles["bend"], 0), np.size(u_his, 1)))
    for icrm in range(np.size(u_his, 1)):
        ui = u_his[:, icrm]
        nodenw = truss["node"].copy()
        nodenw[:, 0] += ui[::3]
        nodenw[:, 1] += ui[1::3]
        nodenw[:, 2] += ui[2::3]

        e_dof_b = np.kron(truss["bars"], np.full((1, 3), 3)) + np.tile([0, 1, 2], (np.size(truss["bars"], 0), 2))
        du = ui[e_dof_b[:, :3]] - ui[e_dof_b[:, 3:6]]
        ex_bar[:, icrm] = (truss["b"] @ ui) / truss["l"].flatten() + 0.5 * np.sum(du ** 2, 1) / (truss["l"].flatten() ** 2)

        for d in range(np.size(angles["bend"], 0)):
            bend : npt.NDArray = angles["bend"][d]
            bd_angle[d, icrm] = fold_ke(nodenw, bend.astype(int))
        
        for f in range(np.size(angles["fold"], 0)):
            fold : npt.NDArray = angles["fold"][f]
            fd_angle[f, icrm] = fold_ke(nodenw, fold.astype(int))

    sx_bar, _, wb = truss["cm"](ex_bar, True)
    rspr_fd = np.zeros(fd_angle.shape)
    e_fold = rspr_fd.copy()
    rspr_bd = np.zeros(bd_angle.shape)
    e_bend = rspr_bd.copy()

    for i in range(np.size(u_his, 1)):
        rspr_fdi, _, e_fold_i = angles["cm_fold"](fd_angle[:, i].reshape((-1, 1)), angles["pf_0"], angles["k_f"].reshape((-1, 1)), truss["l"][np.size(angles["bend"], 0) : np.size(angles["bend"], 0) + np.size(angles["fold"], 0)], True)
        rspr_bdi, _, e_bend_i = angles["cm_bend"](bd_angle[:, i].reshape((-1, 1)), angles["pb_0"], angles["k_b"].reshape((-1, 1)), truss["l"][: np.size(angles["bend"], 0)], True)
        rspr_fd[:, [i]] = rspr_fdi
        e_fold[:, [i]] = e_fold_i
        rspr_bd[:, [i]] = rspr_bdi
        e_bend[:, [i]] = e_bend_i
    
    us_i = np.diag((truss["l"] * truss["a"]).flatten()) @ wb
    br : Bar = {
        "ex" : ex_bar,
        "sx" : sx_bar,
        "us_i" : us_i,
        "us" : np.sum(us_i, 0)
    }

    fd : Fold = {
        "angle" : fd_angle,
        "rm": rspr_fd,
        "uf_i" : e_fold,
        "uf" : np.sum(e_fold, 0)
    }

    bd : Bend = {
        "angle" : bd_angle,
        "rm" : rspr_bd,
        "ub_i" : e_bend,
        "ub" : np.sum(e_bend, 0)
    }

    stat : Stat = {
        "bar" : br,
        "fold" : fd,
        "bend" : bd,
        "pe" : br["us"] + fd["uf"] + bd["ub"]
    }

    return stat
