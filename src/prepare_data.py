import numpy as np
import numpy.typing as npt
import numpy.matlib as npm
from functools import partial
from scipy.optimize import minimize, OptimizeResult # type: ignore
from typing import TypedDict, NotRequired, Callable, Any, Tuple, Union, Optional
from .util.fold_ke import fold_ke
from .util.ogden import ogden
from .util.enhanced_linear import enhanced_linear
from .util.super_linear_bend import super_linear_bend
from .util.globalk_fast_ver import globalk_fast_ver
from .util.dicts import *


def prepare_data(node : npt.NDArray, panel : npt.NDArray, supp : npt.NDArray, load : npt.NDArray, analy_input_opt : AnalyInputOpt) -> Tuple[Truss, Angles, AnalyInputOpt]:
    """Prepares an origami pattern for simulation

    Parameters
    ----------
    node : npt.NDArray
        An nx3 array containing nodal coordinates for the orginial origami pattern before discretization
    panel : npt.NDArray
        Either an nxm array, with n panels and m nodes per panel, or an nx1 array containing lists,
        where each list represents a panel. This format is necessary if not all panels are of the 
        same size, and it can be created by explicitly setting the dtype to "object" when instantiating 
        a non-homogenous numpy array. 
        In either case, a panel is represented by a sequence of nodal indices from the node array, listed in CCW order.
    supp : npt.NDArray
        This is an nx4 array, where each row represents a support condition for a node. The first element of each row should
        be the index of the node that it applies to, and the other three elements in that row represent freedom in the x, y, and z
        directions respectively. The values can be either 0, representing unconstrained motion in one direction, or 1, representing
        a node fixed on that axis. All nodes are assumed to be unconstrained in all axes by default, and only need to be specified if
        they're constrained on one or more axes.
    load : npt.NDArray
        This is structured identically to the supp array, but the last three arguments of every row represent either the load or displacement
        of that node on the x, y, and z axes respectively, depending whether the simulation is set to force or displacement mode.
    analy_input_opt : AnalyInputOpt
        This is a dictionary in which various simulation options can be specified.
        \n1. model_type - This parameter specifies the discretization scheme. The valid options are "n4b5" and "n5b8". The N5B8 discretization scheme is highly recommended, since the N4B5 scheme has a high chance of failing to correctly discretize the shape and is also generally less accurate for simulation.
        \n2. mater_calib - This parameter specifies the calibration method for elemental constitutive behaviour. The user chooses between manual and auto mode. In the manual mode, the constitutive models involved in a bar-and-hinge model need to be specified explicitly using the input fields: bar_cm, rot_spr_bend, rot_spr_fold, a_bar, K_b, and K_f. In the auto mode, the constitutive models are specified implicitly based on actual material properties using the formulas defined in [Filipov et al. 17]. The following input fields are required: mod_elastic, poisson, thickness, and l_scale_factor. The auto mode only applies to N5B8 models.
        \n3. bar_cm - This is a function that defines the constitutive model for bar elements in the following format: Sx,Ct,W = bar_cm(Ex, incl_w). The inputs are the Green-Lagrange strain Exx (Ex) and incl_w, which allows the user to optionally include w or not (if False, None should be returned in place of W), and the three outputs are the 2nd PK stress Sxx (Sx), tangent modulus C (Ct), and strain energy density W (W). The default function is a hyper elastic model (@Ogden) as reported in [Liu and Paulino 17]. To customize, one can write a separate function and pass a function handle to this parameter. 
        \n4. rot_spr_bend - This is a function that defines the constitutive model for bending rotational spring elements in the following format: M,k,P = rot_spr_bend(he,h0,Kp,L0, incl_p). The five inputs are: deformed bending angles θ (he), neutral angles θ0 (h0), stiffness parameter(s) that may differ for each element (Kp), hinge lengths L (Lo), and incl_p, which determines whether P is calculated and returned (if False, None should be returned in place of P). The three outputs shall be resistant moment M (M), tangent rotational modulus k (k), and stored energy ψ (P). Input and output values (except for K_p) are all Nbend×1 arrays, where Nbend denotes the total number of bending hinges. The default function implements an enhanced linear elastic model [Liu and Paulino 17] (@EnhancedLinear) in the manual mode, and a super-linear model [Filipov et al. 17] (@SuperLinearBend) in the auto mode.
        \n5. rot_spr_fold - This function specifies the constitutive model for folding rotational spring elements, following the same format as rot_spr_bend, except that the size of input and output arrays shall be Nf old ×1, where Nf old denotes the total number of folding hinges. The default function is the enhanced linear elastic model (@EnhancedLinear) in both manual and auto modes.
        \n6. a_bar - Areas of bar elements, which could either be a scalar (for uniform area) or a Nbar ×1 array where Nbar denotes the total number of bar elements in the bar-and-hinge model. This is only used in manual mode.
        \n7. K_b - Stiffness parameters for bending rotational spring elements, which could either be a 1×m array (for uniform property) or a Nbend ×m array, where m is the number of required stiffness parameters, which typically equals 1. This is the input Kp for rot_spr_bend.
        \n8. K_f - Stiffness parameters for folding rotational spring elements, following the same format as Kb. This is the input Kp for rot_spr_fold.
        \n9. mod_elastic - Modulus of elasticity E (or Young’s modulus) of the material of the origami structure. This parameter is used in the auto mode.
        \n10. poisson - Poisson’s ratio ν of the material, used in the auto mode.
        \n11. thickness - Thickness t of the origami panels, used in the auto mode.
        \n12. l_scale_factor - Ratio of length scale factor L∗/LF, where LF is the length ofa folding hinge and L∗ is the length scale factor defined in [Filipov et al. 17]. This parameter is used to determine the rotational modulus of folding rotational springs, used in the auto mode. Typical values of L∗/LF is between 1 to 5. The default value is 3.
        \n13. zero_bend - This parameter specifies the neutral angles of bending hinges. User can choose between "flat" and "as_is", or specify user-defined values. This is useful when the initial configuration involves bended panels. The option "as_is" uses the bended shapes of panels as their undeformed states; while the option "flat" always assumes that the neutral angles of bending hinges are π (referring to flat panels). When specifying user-defined values, we use a scalar for uniform neutral angles and a Nbend ×1 array for non-uniform neutral angles.
        \n14. load_type - User can choose between "force" and "displacement".
        \n15. load - In Force mode, the actual applied loads are multiples of the specified load, the scalar multipliers (known as load factor) is determined automatically by a numerical continuation algorithm [Leon et al. 14]. In Displacement mode, the solver attempts the specified amount of displacements in an incremental manner, and stops when the specified amount of displacements is achieved.
        \n16. adaptive_load - This is a function that specifies an adaptive load in the following format: f = load_fun(Node,U,icrm). The inputs are initial nodal coordinates (Node), displacement vector (U), and incremental number (icrm), which can be regarded as pseudo-time. The output is the load vector, which can be either Force or Displacement, but each entry must corresponds to the same degree of freedom as in U. In Displacement mode, the size of each incremental step must be considered within the function. Once specified, the adaptive_load overrides Load.
        \n17. initial_load_factor - Initial load factor for the numerical continuation algorithm [Liu and Paulino 17,Leon et al. 14], used in Force mode.
        \n18. max_incr - Maximum number of increments (Nicrm) for the numerical continuation algorithm, used in Force mode.
        \n19. disp_step - Number of incremental steps to achieve a Displacement load.
        \n20. stop_criterion - A function that specifies a stopping criterion for the numerical simulation based on configurational changes of the origami structure. Use the following format: flag : bool = stop_criterion(node,u,icrm). The function returns True when the stopping criterion is met; otherwise the function should return False.


    Returns
    -------
    truss : Truss
        truss is a dict with the following fields:
        \n1. node - This is a three-column array of size N'node × 3 in the same format as node, but contains additional Steiner points for generalized N5B8 models.
        \n2. bars - Connectivity information of bar elements stored in a Nbar ×2 array. The two entries in a row refer to the indices of the two end nodes of a bar element.
        \n3. trigl - Triangulation information stored in a Ntri ×3 array, where Ntri equals the total number of triangles in the bar-and-hinge model after discretization. The three entries in a row are vertex indices of each triangle.
        \n4. b - Compatibility matrix of the bar frame [Filipov et al. 17] of size Nbar ×Ndof, where Ndof is the total number of degrees of freedom in the system (= 3N'node).
        \n5. l - Initial lengths of bar elements stored in a Nbar ×1 array.
        \n6. fixed_dofs - Indices of fixed degrees of freedom specified by supp.
        \n7. cm - Constitutive model for bar elements, passed from AnalyInputOpt.bar_cm.
        \n8. a - Member areas of bar elements stored in a Nbar ×1 array.
        \n9. u_0 - Initial displacement referring to the unformed configuration. Most of the time, the initial configuration of a structure in a simulation is the undeformed configuration, thus U0 is a Ndof ×1 array of zeros. This parameter is specified outside the PrepareData function, but before the PathAnalysis function is executed. If not specified, the PathAnalysis function assigns truss.u_0 a vector of zeros.
    angles : Angles
        angles is a dict with the following fields:
        \n1. Panel - Same as the aforementioned input parameter Panel.
        \n2. fold - A Nfold × 4 array that contains indices of associated nodes of folding rotational springs. The first two entries define the rotation axis, and the later two refer to off-axis points of the two adjacent triangles in a rotational spring element. See [Liu and Paulino 16] for examples.
        \n3. bend - A Nbend × 4 array that contains indices of associated nodes of bending rotational springs, in the same format as angles.fold.
        \n4. pf_0 - Neutral angles of folding rotational springs stored in a Nfold ×1 array.
        \n5. pb_0 - Neutral angles of bending rotational springs stored in a Nbend ×1 array.
        \n6. cm_bend - Same as AnalyInputOpt.rot_spr_bend.
        \n7. cm_fold - Same as AnalyInputOpt.rot_spr_fold.
        \n8. k_b - Stiffness parameters for bending rotational springs specified for each element in an array of Nbend rows.
        \n9. k_f - Stiffness parameters for folding rotational springs specified for each element in an array of Nf old rows.
    analy_input_opt : AnalyInputOpt
        Returns a populated version of the analy_input_opt argument. This has the same fields as listed above.

    Raises
    ------
    ValueError
        If an incorrect input is given
    """
    put_opt = partial(_put_if_none, analy_input_opt)
    get_opt = partial(_get_or_default, analy_input_opt)
    put_opt("stop_criterion", lambda a, b, c: False)
    put_opt("model_type", "N5B8")
    put_opt("load_type", "force")
    put_opt("max_incr", 100)
    put_opt("initial_load_factor", 0.01)

    if(analy_input_opt.get("model_type") not in ["N5B8", "N4B5"]):
        raise ValueError(f"Model type {analy_input_opt.get('model_type')} is not allowed.")
    
    bend, node, panelctr = _find_bend(panel, node, analy_input_opt.get("model_type"))
    # find bending hinges
    fold, bdry, trigl = _findfdbd(panel, bend)
    # find folding hinges and boundaries, return final triangulation
    bars = np.vstack([bend[:, :2], fold.reshape(-1, (bend.shape)[1])[:, :2], bdry]).astype("int32")
    # define bar elements
    b, l = _direc3d(node, bars)
    if np.size(supp, 0) == 0:
        rs = []
    else: # TODO: there's almost certainly some issues with entangled references. Tons of np.copy()s to add
        rs = np.asarray(np.block([
            np.reshape(np.block([3* supp[:, 0], 3* supp[:, 0] + 1, 3 * supp[:, 0] + 2]), (-1, 1)), 
            np.reshape(supp[:, 1:4].T, (-1, 1))]))
        rs = np.delete(rs, np.where(rs[:, 1] == 0), axis = 0)[:, 0]
    
    a_bar = get_opt("a_bar", 0.1)
    if (type(a_bar) == float):
        # initialize bars array
        a_bar = a_bar * np.ones((np.size(bars, 0), 1))
    
    pf_0 = np.zeros((np.size(fold, 0), 1))
    for i in range(np.size(fold, 0)):
        pf_0[i] = fold_ke(node, fold[i].astype(int))
    
    zero_bend = get_opt("zero_bend", "as_is")
    pb_0 = np.zeros((np.size(bend, 0), 1))

    if type(zero_bend) != str:
        pb_0 += zero_bend
    elif zero_bend == "flat":
        pb_0 += np.pi
    elif zero_bend == "as_is":
        for i in range(np.size(bend, 0)):
            pb_0[i] = fold_ke(node, bend[i].astype("int32"))

    if load.shape[0] != 0:
            m = np.size(node, 0)
            fd = np.zeros((3*m, 1))
            indp = load[:, 0].astype(int)
            fd[3 * indp] = load[:, 1].reshape(-1, 1)
            fd[3 * indp + 1] = load[:, 2].reshape(-1, 1)
            fd[3 * indp + 2] = load[:, 3].reshape(-1, 1)
            analy_input_opt["load"] = fd

    kpb : None | npt.NDArray = None
    # find potential energy constants of bends and folds
    kptb = get_opt("K_b", 0.1)
    if type(kptb) == float and np.size(bend, 0) > 1:
        kpb = np.asarray(npm.repmat(kptb, np.size(bend, 0), 1))
    elif type(kptb) == np.ndarray:
        kpb = kptb
    
        
    if kpb is not None:
        kpf = get_opt("K_f", 0.1 * kpb[0])
        if np.size(kpf, 0) == 1 and np.size(fold, 0) > 1:
            kpf = np.asarray(npm.repmat(kpf, np.size(fold, 0), 1))

    if (analy_input_opt.get("model_type") == "N4B5"):
        
        truss : Truss = {
            "node" : node,
            "bars" : bars,
            "trigl" : trigl.astype(int),
            "b" : b,
            "l" : l.reshape((-1,1)),
            "cm" : get_opt("bar_cm", lambda Ex, incl: ogden(Ex, 1e4, incl)),
            "fixed_dofs" : np.unique(rs),
            "a" : a_bar.reshape((-1, 1)),
            "u_0": get_opt("u_0", np.zeros((np.size(node, 0), 1)))
        }

        angles : Angles = {
            "cm_fold" : get_opt("rot_spr_fold", lambda he,h0,kb,l0,incl_espr: enhanced_linear(he, h0, kb, l0, 45, 315, incl_espr)),
            "cm_bend" : get_opt("rot_spr_bend", lambda he,h0,kb,l0,incl_espr: enhanced_linear(he, h0, kb, l0, 45, 315, incl_espr)),
            "fold" : fold,
            "bend" : bend,
            "k_b" : kpb,
            "k_f" : kpf ,
            "pf_0" : pf_0,
            "pb_0" : pb_0,
            "panel" : panel
        }
        

    elif analy_input_opt.get("model_type") == "N5B8":
        
        mater_calib = get_opt("mater_calib", "auto")

        if (mater_calib == "manual"):
            truss : Truss = {
                "node" : node,
                "bars" : bars,
                "trigl" : trigl.astype(int),
                "b" : b,
                "l" : l.reshape((-1, 1)),
                "fixed_dofs" : np.unique(rs),
                "cm" : get_opt("bar_cm", lambda x, incl: ogden(x, 1e4, incl)),
                "a" : a_bar.reshape((-1, 1)),
                "u_0" : None
            }

            angles : Angles = {
                "fold" : fold,
                "bend" : bend,
                "pf_0" : pf_0,
                "pb_0" : pb_0,
                "panel" : panel,
                "cm_bend" : get_opt("rot_spr_bend", lambda he,h0,kb,l0,incl_espr: enhanced_linear(he, h0, kb, l0, 45, 315, incl_espr)),
                "cm_fold" : get_opt("rot_spr_fold", lambda he,h0,kb,l0,incl_espr: enhanced_linear(he, h0, kb, l0, 45, 315, incl_espr)),
                "k_b" : kpb,
                "k_f" : kpf
            }
            

        else: 
            ey = get_opt("mod_elastic", 1e9)
            nv = get_opt("poisson", 0.33)
            thck = get_opt("thickness", 0.15e-2)
            a_bar = np.zeros((np.size(bars, 0),))
            kpb = np.zeros((np.size(bend, 0), 1))
            kpf = np.zeros((np.size(fold, 0), 1))
            g = ey * (thck ** 3) / (12 * (1 - (nv ** 2)))
            lf = l[np.arange(np.size(kpb), np.size(kpb) + np.size(kpf))]
            ls = get_opt("l_scale_factor", 2 * np.average(lf))
            kl = g * ((1/np.max(ls)))
            km = np.power((lf / thck), 1 / 3) * (0.55 * g)
            kpf = (1 / (1 / kl + 1 / km)) / lf

            panelctr = panelctr.astype("int32")
            # confirmed sameness up to k_f
            for j in range(np.size(panelctr, 0)):
                abarj, kpbj = _get_material(node, np.array(panel[j], dtype=int), panelctr[j], ey, nv, thck, fold.astype(int), bend.astype(int), bdry.astype(int))
                a_bar[abarj[:, 0].astype("int32")] = a_bar[abarj[:, 0].astype("int32")] + abarj[:, 1]
                if kpbj.shape[0] != 0:
                    kpb[kpbj[:, 0].astype("int32")] = (kpb[kpbj[:, 0].astype("int32")] + np.reshape(kpbj[:, 1], (-1, 1)))
            
            truss : Truss = {
                "node" : node,
                "bars" : bars,
                "trigl" : trigl.astype(int),
                "b" : b,
                "l" : l.reshape((-1, 1)),
                "fixed_dofs" : np.unique(rs),
                "cm" : get_opt("bar_cm", lambda ex, incl: ogden(ex, ey, incl)),
                "a" : a_bar.reshape((-1, 1)),
                "u_0" : None
            }

            angles : Angles = {
                "fold" : fold,
                "bend" : bend,
                "pf_0" : pf_0,
                "pb_0" : pb_0,
                "panel" : panel,
                "cm_bend" : get_opt("rot_spr_bend", lambda he, h0, kp, l0, include_espr: super_linear_bend(he, h0, kp, l0, include_espr)),
                "cm_fold" : get_opt("rot_spr_fold", lambda he,h0,kb,l0, incl_espr: enhanced_linear(he, h0, kb, l0, 45, 315, incl_espr)),
                "k_b" : kpb,
                "k_f" : kpf
            }

    else:
        raise ValueError()
    return truss, angles, analy_input_opt

def _get_material(node, lst, indexctr, e, nv, t, fold, bend, bdry):

    fold = np.sort(fold[:, : 2], 1)
    nf = np.size(fold, 0)
    bend = np.sort(bend[:, : 2], 1)
    nb = np.size(bend, 0)
    bdry = np.sort(bdry, 1)
    g = e * (t ** 3) / (12 * (1 - (nv ** 2)))
    pairs = np.sort(np.vstack([lst, np.roll(lst, np.size(lst) - 1)]), 0).T
    lf = np.sqrt(np.sum(np.power(node[pairs[:, 1]] - node[pairs[:, 0]], 2), 1))

    if (np.size(lst)) == 3:
        s = 0.5 * np.linalg.norm(np.cross(node[lst[1]] - node[lst[0]], node[lst[2]] - node[lst[0]]))
        a = t * 2 * s / (np.sum(lf) * (1 - nv)) # t * s introduces noticeable error in matlab
        _, indfd, _ = _intersect_2d(fold.astype(int), pairs.astype(int))
        if (np.size(indfd) < 3):
            _, inddd, _ = _intersect_2d(bdry, pairs)
        else:
            inddd = np.array([])
        kpbj = np.empty((0,))
        abarj = np.hstack([np.hstack([indfd + nb, inddd + nf + nb]).reshape(-1, 1), a * np.ones((np.size(lst), 1))])

    elif np.size(lst) == 4:
        _, indfd, efd = _intersect_2d(fold.astype(int), pairs.astype(int))
        spoke = np.sort(np.hstack((lst.reshape((1, 4)).T, np.ones((np.size(lst), 1)) * indexctr)), 1)
        lb = np.sqrt(np.sum(np.power(node[lst] - node[indexctr], 2), 1))
        _, indbd, _ = _intersect_2d(bend.astype(int), spoke.astype(int))
        dsl = lb + lb[[2, 3, 0, 1]]
        kb = np.divide(g * np.power((dsl / t), 1 / 3), dsl)
        kpbj = np.block([indbd.reshape((-1, 1)), kb.reshape(-1, 1)])
        w = np.sum(lf[[0, 2]]) / 2
        h = np.sum(lf[[1, 3]]) / 2
        af = np.zeros((4, 1))
        af[[0, 2]] = t * abs((h ** 2) - nv * (w ** 2)) / (2 * h * (1 - (nv ** 2)))
        af[[1, 3]] = t * abs((w ** 2) - nv * (h ** 2)) / (2 * w * (1 - (nv ** 2)))
        ab = np.ones((4, 1)) * (t * nv * (((h ** 2) + (w ** 2)) ** 1.5) / (2 * h * w * (1 - (nv ** 2))))
        if np.size(indfd) == np.size(lst):
            abarj = np.hstack([np.vstack([indbd.reshape((-1, 1)), (indfd + nb).reshape((-1, 1))]), np.vstack([ab, af])])
        elif np.size(indfd) < np.size(lst):
            _, inddd, edd = _intersect_2d(bdry, pairs)
            abarj = np.hstack([np.vstack([indbd.reshape((-1, 1)), (indfd + nb).reshape((-1, 1)), (inddd + nf + nb).reshape((-1, 1))]), np.vstack([ab.reshape((-1, 1)), af[efd].reshape((-1, 1)), af[edd].reshape((-1, 1))])])
        else:
            raise ValueError()
    elif np.size(lst) > 4:
        _, indfd, _ = _intersect_2d(fold.astype(int), pairs.astype(int))
        spoke = np.sort(np.hstack((lst.reshape((1, -1)).T, np.ones((np.size(lst), 1)) * indexctr)), 1)
        sa = np.subtract(node[lst], node[indexctr, :])
        lb = np.sqrt(np.sum(np.power(sa, 2), 1))
        sac = np.cross(sa, node[lst[list(range(1, np.size(lst))) + [0]]] - node[indexctr])
        s = 0.5 * np.sum(np.sqrt(np.sum(np.power(sac, 2), 1)))
        a = (t * 2 * s) / ((np.sum(lf) + np.sum(lb)) * (1 - nv))    # t * s calculation is imprecise in matlab
        _, indbd, _ = _intersect_2d(bend.astype(int), spoke.astype(int))
        if np.size(indfd) == np.size(lst):
            abarj = np.hstack([np.hstack([indbd, indfd + nb]).reshape(1, -1), a * np.ones((2 * np.size(lst), 1))])
        elif np.size(indfd) < np.size(lst):
            _, inddd, _ = _intersect_2d(bdry, pairs.astype(int))
            abarj = np.hstack([np.hstack([indbd, indfd + nb, inddd + nb + nf]).reshape(1, -1).T, a * np.ones((2 * np.size(lst), 1))])
        else:
            raise ValueError()
        np.divide(sa.T, lb).T
        D = np.matmul(sa, sa.T)
        id = np.argmin(D, 0)
        dsl = lb + lb[id]
        kb = np.divide(g * ((dsl / t) ** (1 / 3)), dsl)
        kpbj = np.vstack([indbd, kb]).T
    else:
        raise ValueError()
    return abarj, kpbj

def _put_if_none(d, opt, default):
    if d.get(opt) == None:
        d[opt] = default

def _get_or_default(d : AnalyInputOpt, opt, default):
    out = d.get(opt)
    return default if out == None else out

# panel is n x 1 made of lists (this is necessary to make it non-rectangular)
def _find_bend(panel : npt.NDArray, node : npt.NDArray, model_type : Union[str, None]) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    if model_type == "N4B5":
        bend = _nan(3 * panel.shape[0], 4)
        cntb = 0
        for i in range(panel.shape[0]):
            p = np.array(panel[i])
            if np.size(p, axis = 0) == 4:
                l1 = np.linalg.norm(node[p[0], :] - node[p[2], :])
                l2 = np.linalg.norm(node[p[3], :] - node[p[1], :])
                lclbend = []
                if l1 > l2: lclbend = [1, 3, 0, 2]
                else: lclbend = [0, 2, 1, 3]
                cntb += 1
                bend[cntb, :] = p[np.array(lclbend)]
            elif np.size(p, axis = 0) > 4:
                bi_edge = _divide_polygon(node[p])     # might need to ensure that bi_edge is 2D
                if (np.size(p, axis = 0) - 3) == np.size(bi_edge, axis = 0):
                    bendlcl = np.zeros((np.size(bi_edge, axis = 0), 4))
                else:
                    raise ValueError("Impossible Triangulation!")
                pp = np.arange(0, np.size(p, axis = 0)).T
                total_links = np.vstack([np.array([np.roll(pp, 1, axis = 0), pp]).T, bi_edge])
                comm = np.zeros((np.size(pp), np.size(total_links, axis = 0)))
                for j in range(np.size(total_links, axis = 0)):
                    comm[total_links[j, :], j] = 1
                g_p = np.matmul(comm, comm.T)
                for k in range(np.size(bi_edge, axis = 0)):
                    myab = np.ndarray.flatten(np.argwhere(g_p[bi_edge[k, 0], :] + g_p[bi_edge[k, 1], :] == 2))
                    bendlcl[k, :] = np.hstack((bi_edge[k], myab))
                bend[cntb : cntb + np.size(bi_edge, axis = 0)] = p[bendlcl.astype("int32")]
                cntb += np.size(bi_edge, axis = 0)
        
        bend = np.delete(bend, np.any(np.isnan(bend), axis = 1), axis = 0)
        panelctr = np.array([])
        return (bend, node, panelctr)
    else: #TODO: node is incorrectly converted
        nn = np.size(node, axis = 0) - 1
        bend = _nan(6 * np.size(panel, 0), 4)
        panelctr = _nan(np.size(panel, 0), 1)
        cntb = -1
        cntc = 0
        node = np.vstack((node, np.full((np.size(panel, 0), np.size(node, 1)), np.nan)))
        for i in range(np.size(panel, 0)):
            p = np.array(panel[i], dtype="int64")
            if np.size(p) == 4:
                cntc += 1
                L1 = np.linalg.norm(node[p[0]] - node[p[2]])
                L2 = np.linalg.norm(node[p[3]] - node[p[1]]) 
                m = np.array(node[p[2]] - node[p[0]]).T / L1
                n = np.array(node[p[3]] - node[p[1]]).T / L2
                # TODO: Revisit this calculation- it's comparable but noticeably off from the matlab calc
                coeff, *_ = np.linalg.lstsq(np.array([m, -n]).T.astype('float64'), (node[p[1]] - node[p[0]]).astype('float64'), rcond=None)
                if L1 < L2: ctrnode = node[p[0]] + m.T * coeff[0]
                else: ctrnode = node[p[1]] + n.T * coeff[1]
                # If 1,2,3,4 are co-planar, the two results are identical;
                # if they are not, the central node is placed such that the
                # panel is bended along the shorter diagonal.
                node[nn + cntc] = ctrnode
                panelctr[i] = nn + cntc
                for k in range(np.size(p)):
                    cntb += 1
                    ref1 = np.mod(k - 1, np.size(p))
                    ref2 = np.mod(k + 1, np.size(p))
                    bend[cntb] = [nn + cntc, p[k], p[ref1], p[ref2]]
            elif np.size(p) > 4:
                cntc += 1
                ctrnode = _find_opt_center(node[p])
                node[nn + cntc] = ctrnode      #TODO: this exceeds teh size of node, need to resize. Potentially just add a node for each panel
                panelctr[i] = nn + cntc
                for k in range(np.size(p)):
                    cntb += 1
                    ref1 = np.mod(k - 1, np.size(p))
                    ref2 = np.mod(k + 1, np.size(p))
                    bend[cntb] = [nn + cntc, p[k], p[ref1], p[ref2]]
        bend = np.delete(bend, np.isnan(bend[:, 1]), axis = 0)
        node = node[ : nn + cntc + 1]
        return (bend, node, panelctr)

def _findfdbd(panel, bend):
    # triangularization
    if (type(panel[0]) == list):
        panel_size_func = np.vectorize(len)
        panel_size = panel_size_func(panel).astype(int)
    else:
        panel_size = (np.ones((np.size(panel, 0), 1)) * np.size(panel, 1)).astype(int)
    panels_3 = np.size(panel_size[panel_size == 3])
    ptri = np.empty((panels_3, 3), dtype=object)
    flg = np.where(panel_size == 3)
    for i in range(panels_3):
        if type(panel[flg[i]][0]) == list:
            ptri[i] = panel[flg[i]][0]
        else:
            ptri[i] = panel[flg[i]]
    ptri = ptri.astype(int)
    trigl_raw = np.vstack([bend[:, [0, 2, 1]], ptri.reshape((-1, 3))])  # there's no way these share dimensions
    trigl_raw_sort = np.sort(trigl_raw, 1)
    trigl, uniqidx = np.unique(trigl_raw_sort, axis = 0, return_index = True)
    # make connectivity matrix
    comm = np.zeros((np.max(np.max(trigl.astype("int32"))) + 1, np.size(trigl, 0)))
    for i in range(np.size(trigl, 0)):
        comm[trigl[i].astype('int32'), i] = 1
    # find fold lines
    Ge = np.matmul(comm.T, comm)
    mf, me = np.nonzero(np.triu(Ge == 2)) # find triangular meshes w/ 2 common nodes
    fold = np.zeros((np.size(mf), 4))
    for i in range(np.size(mf)):
        # find shared vertices and the vertices of neighboring triangles
        link, ia, ib = np.intersect1d(trigl[mf[i]], trigl[me[i]], return_indices = True)
        oftpa = np.setdiff1d(np.arange(3), ia)
        oftpb = np.setdiff1d(np.arange(3), ib)

        # check ordering of nodes
        wrapverts = np.block([trigl_raw[uniqidx[mf[i]]], trigl_raw[uniqidx[mf[i]]]])
        ismf = np.nonzero(np.bitwise_and((wrapverts[:-1] == link[0]), (wrapverts[1:] == link[1])))
        if not np.size(ismf) == 0:
            fold[i] = np.block([link, trigl[mf[i], [oftpa]], trigl[me[i], oftpb]])
        else:
            wrapverts = np.block([trigl_raw[uniqidx[me[i]]], trigl_raw[uniqidx[me[i]]]])
            isme = np.nonzero(np.bitwise_and((wrapverts[:-1] == link[0]), (wrapverts[1:] == link[1])))
            if not np.size(isme) == 0:
                fold[i] = np.block([link, trigl[me[i], oftpb], trigl[mf[i], oftpa]])
            else:
                print("WARNING: could not find correct ordering. Hit enter to continue...")
                input()
        pass
    
    fd_and_bd = np.sort(fold[:, :2], 1)
    only_bd = np.sort(bend[:, :2], 1)
    _, ibd, _ = _intersect_2d(fd_and_bd, only_bd)
    fold = np.delete(fold, ibd, axis = 0)

    # look for boundaries
    edge = np.sort(np.block([[trigl[:, 0], trigl[:, 1], trigl[:, 2]],[trigl[:, 1], trigl[:, 2], trigl[:, 0]]]).T, 1)
    u, n = np.unique(edge, axis = 0, return_inverse=True)
    counts = np.bincount(n)
    bdry = u[counts==1]
    return fold, bdry, trigl


def _nan(*shape):
    arr = np.empty(shape)
    arr.fill(np.nan)
    return arr


def _divide_polygon(poly_coord : npt.NDArray) -> npt.NDArray:
    if (poly_coord.shape[0]) <= 3:
        return np.array([])
    else:
        G = np.triu(np.ones((poly_coord.shape[0], poly_coord.shape[0])), 2)
        G[0, -1] = 0
        I, J = np.nonzero(G)
        L2 = np.sum(np.power(poly_coord[I, :] - poly_coord[J, :], 2), axis = 1)     # this seems potentially incorrect
        ind_min = np.argmin(L2, axis = 0)    # row indices of minimum in each col
        bi_edge = np.sort(np.array([I[ind_min], J[ind_min]]).T, axis = 0)
        T1 = np.concatenate((np.arange(0, bi_edge[0] + 1), np.arange(bi_edge[1], poly_coord.shape[0])))
        T2 = np.arange(bi_edge[0], bi_edge[1] + 1)
        return np.vstack([bi_edge.reshape((-1, 2)),
                        T1[_divide_polygon(poly_coord[T1, :])].reshape((-1, 2)),
                        T2[_divide_polygon(poly_coord[T2, :])].reshape((-1, 2))])

def _find_opt_center(poly_coord : npt.NDArray):
    G = np.triu(np.ones((np.size(poly_coord, 0), np.size(poly_coord, 0))), 2)
    G[0, -1] = 0
    I, J = np.nonzero(G)
    L2 = np.sum(np.power(poly_coord[J] - poly_coord[I], 2), axis = 1)

    obj_fun = lambda xc: np.sum(np.divide(np.sqrt(np.sum(np.power(np.cross((poly_coord[J] - xc), (poly_coord[I] - xc)).T, 2), 0)), np.power(L2, 1)), 0)
    idmin = np.argmin(L2)
    XC01 = (poly_coord[I[idmin]] + poly_coord[J[idmin]]) / 2
    XC02 = np.sum(poly_coord, 0) / np.size(poly_coord, 0)

    res1 : OptimizeResult = minimize(obj_fun, XC01, method = "BFGS")
    res2 : OptimizeResult = minimize(obj_fun, XC02, method = "BFGS")

    xc = res1.x if res1.fun <= res2.fun else res2.x
    print(f"MinObj (MidPoint = {res1.fun}, CenterPoint = {res2.fun})")
    return xc

# taken from https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
def _intersect_2d(A, B):
    A = A.copy()
    B = B.copy()
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
        'formats':ncols * [A.dtype]}
    C, aInd, bInd = np.intersect1d(A.view(dtype), B.view(dtype), return_indices=True)
    C = C.view(A.dtype).reshape(-1, ncols)
    return C, aInd, bInd

def _direc3d(node, ele):
    ne = np.size(ele, 0)
    nn = np.size(node, 0)
    D = np.vstack([node[ele[:, 1], 0] - node[ele[:, 0], 0], node[ele[:, 1], 1] - node[ele[:, 0], 1], node[ele[:, 1], 2] - node[ele[:, 0], 2]]).T
    L = np.sqrt(np.power(D[:, 0], 2) + np.power(D[:, 1], 2) + np.power(D[:, 2], 2))
    D = np.array([np.divide(D[:, 0], L), np.divide(D[:, 1], L), np.divide(D[:, 2], L)]).T
    B = np.zeros((ne, 3 * nn))
    B[npm.repmat(np.arange(ne).reshape(1, ne).T, 1, 6).flatten(), 
            np.array([3 * ele[:, 0] + 1 - 1, 3 * ele[:, 0] + 2- 1, 3 * ele[:, 0] + 3- 1, 3 * ele[:, 1] + 1- 1, 3 * ele[:, 1] + 2- 1, 3 * ele[:, 1] + 3- 1]).T.flatten()
        ] = np.block([D, -D]).flatten()
    B = -B
    return B, L




# if __name__ == "__main__":
#     # testing
#     truss, angle, analy_input_opt = test_single_plane(False)
    # pass
