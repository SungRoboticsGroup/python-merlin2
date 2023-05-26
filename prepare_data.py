import numpy as np
import numpy.typing as npt
import numpy.matlib as npm
from functools import partial
from scipy.optimize import minimize, OptimizeResult
from scipy.sparse import csr_matrix
from typing import TypedDict, NotRequired, Callable, Any, Tuple, Union, Optional
from .fold_ke import fold_ke
from .odgen import ogden
from .enhanced_linear import enhanced_linear
from .super_linear_bend import super_linear_bend

# TODO: Find out if bar_cm should have just an NDArray instead
class AnalyInputOpt(TypedDict):
    model_type : NotRequired[str]   
    mater_calib : NotRequired[str]  
    bar_cm : NotRequired[Callable[[npt.NDArray, bool], Tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray]]]]
    rot_spr_bend : NotRequired[Callable[[npt.ArrayLike, npt.ArrayLike, Any, npt.ArrayLike], Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]]]
    rot_spr_fold : NotRequired[Callable[[npt.ArrayLike, npt.ArrayLike, Any, npt.ArrayLike], Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]]]
    a_bar : NotRequired[Union(npt.ArrayLike, float)]
    K_b : NotRequired[npt.ArrayLike]
    K_f : NotRequired[npt.ArrayLike]
    mod_elastic : NotRequired[float]
    poisson : NotRequired[float]
    thickness : NotRequired[float]
    l_scale_factor : NotRequired[float]
    zero_bend : NotRequired[Union(str, float, npt.ArrayLike)]
    load_type : NotRequired[str]
    load : NotRequired[npt.NDArray]
    adaptive_load : NotRequired[Callable[[npt.ArrayLike, npt.ArrayLike, float], npt.ArrayLike]]
    initial_load_factor : NotRequired[float]
    max_incr : NotRequired[int]
    disp_step : NotRequired[int]
    stop_criterion : NotRequired[Callable[[npt.ArrayLike, npt.ArrayLike, float], bool]]

class Truss(TypedDict):
    node : npt.ArrayLike
    bars : npt.ArrayLike
    trigl : npt.ArrayLike
    b : npt.ArrayLike
    l : npt.ArrayLike
    fixed_dofs : npt.ArrayLike
    cm : Callable[[Any, Any, Any], Any]
    a : npt.ArrayLike
    u_0 : npt.ArrayLike

class Angles(TypedDict):
    panel : npt.ArrayLike
    fold : npt.ArrayLike
    bend : npt.ArrayLike
    pf_0 : npt.ArrayLike
    pb_0 : npt.ArrayLike
    cm_bend : Callable[[npt.ArrayLike, npt.ArrayLike, Any, npt.ArrayLike], Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]]
    cm_fold : Callable[[npt.ArrayLike, npt.ArrayLike, Any, npt.ArrayLike], Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]]
    k_b : npt.ArrayLike
    k_f : npt.ArrayLike  


# analy_input_opt contains the following fields:
#     'ModelType': 'N4B5' or 'N5B8'
#     'MaterCalib': 'auto' or 'manual'
#     'BarCM': Constitutive model of bar elements
#     'Abar': Area of bars (manual mode)
#     'RotSprBend': Constitutive model of bending elements (manual mode)
#     'RotSprFold': Constitutive model of folding elements (manual mode)
#     'Kb': Initial modulus of bending hinges
#     'Kf': Initial modulus of folding hinges
#     'ModElastic': Modulus of Elasticy of bar material
#     'Poisson': Poisson's ratio
#     'Thickness': Panel thickness
#     'LScaleFactor': Ratio of length scale factor (Ls) over hinge length LF
#     'ZeroBend': 'AsIs' - as the geometry is defined, 
#                 'Flat' - enforce neutral angle at pi (flat configuration)
#                  value - provide specific values, scalar for uniform
#                  assignment, vector for differential assignment 
#     'LoadType': 'Force' or 'Displacement'
#     'Load': Loading condition
#     'AdaptiveLoad': Function handle for adaptive loading
#     'InitialLoadFactor': Initial load factor for MGDCM ('Force' mode)
#     'MaxIcr': Maximum number of increments ('Force' mode)
#     'DispStep': Number of displacement increments ('Displacement' mode)
#     'StopCriterion': Function handle for specific stopping criterion.

def prepare_data(node : npt.ArrayLike, panel : npt.ArrayLike, supp : npt.ArrayLike, load : npt.ArrayLike, analy_input_opt : AnalyInputOpt):
    check_opt = partial(_check_or_default, analy_input_opt)
    get_opt = partial(_get_or_default, analy_input_opt)
    check_opt("stop_criterion", lambda a, b, c: False)
    check_opt("model_type", "N5B8")
    check_opt("load_type", "force")
    check_opt("max_incr", 100)
    check_opt("initial_load_factor", 0.01)

    if(analy_input_opt["model_type"] not in ["N5B8", "N4B5"]):
        raise ValueError(f"Model type {analy_input_opt['model_type']} is not allowed.")
    
    bend, node, panelctr = _find_bend(panel, node, analy_input_opt["model_type"])
    # find bending hinges
    fold, bdry, trigl = _findfdbd(panel, bend)
    # find folding hinges and boundaries, return final triangulation
    bars = np.block([bend[:, :1], fold[:, :1], bdry])
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
        Abar = Abar * np.ones((np.size(bars, 0), 1))
    
    pf_0 = np.zeros((np.size(fold, 0), 1))
    for i in range(np.size(fold, 0)):
        pf_0[i] = fold_ke(node, fold[i])
    
    zero_bend = get_opt("zero_bend", "as_is")
    pb_0 = np.zeros((np.size(bend, 0), 1))

    if type(zero_bend) != str:
        pb_0 += zero_bend
    elif zero_bend == "flat":
        pb_0 += np.pi
    elif zero_bend == "as_is":
        for i in range(np.size(bend, 0)):
            pb_0[i] = fold_ke(node, bend[i])

    if load != None:
            m = np.size(node, 0)
            fd = np.zeros((3*m, 1))
            indp = load[:, 1]
            fd[3 * indp - 3] = load[:, 1]
            fd[3 * indp - 2] = load[:, 2]
            fd[3 * indp - 1] = load[:, 3]
            analy_input_opt["load"] = fd

    # find potential energy constants of bends and folds
    kpb = get_opt("K_b", 0.1)
    if np.size(kpb, 0) == 1 and np.size(bend, 0) > 1:
        kpb = np.asarray(npm.repmat(kpb, np.size(bend, 0), 1))
        
    kpf = get_opt("K_f", 0.1 * kpb[1])
    if np.size(kpf, 0) == 1 and np.size(fold, 0) > 1:
        kpf = np.asarray(npm.repmat(kpf, np.size(fold, 0), 1))

    if (analy_input_opt.get("model_type") == "N4B5"):
        
        truss : Truss = {}
        truss["cm"] = get_opt("bar_cm", lambda Ex, incl: ogden(Ex, 1e4, incl))
        truss["node"] = node
        truss["bars"] = bars
        truss["trigl"] = trigl
        truss["b"] = b
        truss["l"] = l
        truss["fixed_dofs"] = np.unique(rs)
        truss["a"] = a_bar

        angles : Angles = {}
        angles["cm_bend"] = get_opt("rot_spr_bend", lambda he,h0,kb,l0: enhanced_linear(he, h0, kb, l0, 45, 315))
        angles["cm_fold"] = get_opt("rot_spr_fold", lambda he,h0,kb,l0: enhanced_linear(he, h0, kb, l0, 45, 315))
        angles["fold"] = fold
        angles["bend"] = bend
        angles["k_b"] = kpb
        angles["k_f"] = kpf 
        angles["pf_0"] = pf_0
        angles["pb_0"] = pb_0
        angles["panel"] = panel

    elif analy_input_opt["model_type"] == "N5B8":
        
        truss : Truss = {}
        truss["node"] = node
        truss["bars"] = bars
        truss["trigl"] = trigl
        truss["b"] = b
        truss["l"] = l
        truss["fixed_dofs"] = np.unique(rs)

        angles : Angles = {}
        
        angles["fold"] = fold
        angles["bend"] = bend
        angles["pf_0"] = pf_0
        angles["pb_0"] = pb_0
        angles["panel"] = panel

        mater_calib = get_opt("mater_calib", "auto")

        if (mater_calib == "manual"):
            truss["cm"] = get_opt("bar_cm", lambda x: ogden(x, 1e4))
            truss["a"] = a_bar

            angles["cm_bend"] = get_opt("rot_spr_bend", lambda he,h0,kb,l0: enhanced_linear(he, h0, kb, l0, 45, 315))
            angles["cm_fold"] = get_opt("rot_spr_fold", lambda he,h0,kb,l0: enhanced_linear(he, h0, kb, l0, 45, 315))

            angles["k_b"] = kpb
            angles["k_f"] = kpf

        else: 
            ey = get_opt("mod_elastic", 1e9)
            nv = get_opt("poisson", 0.33)
            thck = get_opt("thickness", 0.15e-2)
            check_opt("bar_cm", lambda ex: ogden(ex, ey))
            a_bar = np.zeros((np.size(bars, 0), 1))
            kpb = np.zeros((np.size(bend, 0), 1))
            kpf = np.zeros((np.size(fold, 0), 1))
            g = ey * (thck ** 3) / (12 * (1 - (nv ** 2)))
            lf = l[np.arange(np.size(kpb) + 1, np.size(kpb) + np.size(kpf))]
            ls = get_opt("l_scale_factor", 2 * np.average(lf))
            kl = g * ((1/np.max(ls)))
            km = np.power(((0.55 * g / thck) * lf), 1 / 3)
            kpf = (1 / (1 / kl + 1 / km)) / lf

            for j in range(np.size(panelctr)):
                abarj, kpbj = _get_material(node, panel[j], panelctr[j], ey, nv, thck, fold, bend, bdry)
                a_bar[abarj[:, 0]] = a_bar[abarj[:, 0]] + abarj[:, 1]
                if kpbj.shape[0] == 0:
                    kpb[kpbj[:, 0]] = kpb[kpbj[:, 0]] + kpbj[:, 1]
            
            truss["a"] = a_bar
            angles["cm_bend"] = get_opt("rot_spr_bend", lambda he, h0, kp, l0, include_espr: super_linear_bend(he, h0, kp, l0, include_espr))
            angles["cm_fold"] = get_opt("rot_spr_fold", lambda he,h0,kb,l0: enhanced_linear(he, h0, kb, l0, 45, 315))
            angles["k_b"] = kpb
            angles["k_f"] = kpf
    return truss, angles, analy_input_opt
    


def _get_material(node, lst, indexctr, e, nv, t, fold, bend, bdry):
    fold = np.sort(fold[:, : 2], 1)
    nf = np.size(fold, 0)
    bend = np.sort(bend[:, : 2], 1)
    nb = np.size(bend, 0)
    bdry = np.sort(bdry, 1)
    g = e * (t ** 3) / (12 * (1 - (nv ** 2)))
    pairs = np.sort(np.array([lst, lst[np.block([np.arange(1, np.size(lst)), 0])]]), 0).T
    lf = np.sqrt(np.sum(np.power(node[pairs[:, 1]] - node[pairs[:, 0]], 2), 1))


    if (np.size(lst)) == 3:
        s = 0.5 * np.linalg.norm(np.cross(node[lst[1]] - node[lst[0]], node[lst[2]] - node[lst[0]]))
        a = t * 2 * s / (np.sum(lf) * (1 - nv)) # t * s introduces noticeable error in matlab
        _, indfd, _ = _intersect_2d(fold, pairs)
        if (np.size(indfd) < 3):
            _, inddd, _ = _intersect_2d(bdry, pairs)
        else:
            inddd = []
        kpbj = []
        abarj = np.block([[indfd + nb, inddd + nf + nb], [a * np.ones((np.size(lst), 1)).T]]).T
    elif np.size(lst) == 4:
        _, indfd, efd = _intersect_2d(fold, pairs)
        spoke = np.sort(np.hstack((lst.reshape((1, 4)).T, np.ones((np.size(lst), 1)) * indexctr)), 1)
        lb = np.sqrt(np.sum(np.power(node[lst] - node[indexctr], 2), 1))
        _, indbd, _ = _intersect_2d(bend, spoke)
        dsl = lb + lb[[2, 3, 0, 1]]
        kb = np.divide(g * np.power((dsl / t), 1 / 3), dsl)
        kpbj = np.block([indbd, kb]).T
        w = np.sum(lf[[0, 2]]) / 2
        h = np.sum(lf[[1, 3]]) / 2
        af = np.zeros((4, 1))
        af[[0, 2]] = t * abs((h ** 2) - nv * (w ** 2)) / (2 * h * (1 - (nv ** 2)))
        af[[1, 3]] = t * abs((w ** 2) - nv * (h ** 2)) / (2 * w * (1 - (nv ** 2)))
        ab = np.ones((4, 1)) * (t * nv * (((h ** 2) + (w ** 2)) ** 1.5) / (2 * h * w * (1 - (nv ** 2))))
        if np.size(indfd) == np.size(lst):
            abarj = np.block([[indbd, indfd + nb], [ab, af]]).T
        elif np.size(indfd) < np.size(lst):
            _, inddd, edd = _intersect_2d(bdry, pairs)
            abarj = np.block([[indbd, indfd + nb, inddd + nf + nb], [ab, af[efd], af[edd]]]).T
    elif np.size(lst) > 4:
        _, indfd, _ = _intersect_2d(fold, pairs)
        spoke = np.sort(np.hstack((lst.reshape((1, 5)).T, np.ones((np.size(lst), 1)) * indexctr)), 1)
        sa = np.subtract(node[lst], node[indexctr])
        lb = np.sqrt(np.sum(np.power(sa, 2), 1))
        sac = np.cross(sa, node[lst[list(range(1, np.size(lst))) + [0]]] - node[indexctr])
        s = 0.5 * np.sum(np.sqrt(np.sum(np.power(sac, 2), 1)))
        a = (t * 2 * s) / ((np.sum(lf) + np.sum(lb)) * (1 - nv))    # t * s calculation is imprecise in matlab
        _, indbd, _ = _intersect_2d(bend, spoke)
        if np.size(indfd) == np.size(lst):
            abarj = np.block([[indbd, indfd + nb], [a * np.ones((2 * np.size(lst), 1)).T]]).T
        elif np.size(indfd) < np.size(lst):
            _, inddd, _ = _intersect_2d(bdry, pairs)
            abarj = np.block([[indbd, indfd + nb, inddd + nb + nf], [a * np.ones((2 * np.size(lst), 1)).T]]).T
        np.divide(sa.T, lb).T
        D = np.matmul(sa, sa.T)
        id = np.argmin(D, 0)
        dsl = lb + lb[id]
        kb = np.divide(g * ((dsl / t) ** (1 / 3)), dsl)
        kpbj = np.block([indbd, kb]).T
    return abarj, kpbj

def _check_or_default(d, opt, default):
    if d.get(opt) == None:
        d[opt] = default

def _get_or_default(d : AnalyInputOpt, opt, default):
    out = d.get(opt)
    return default if out == None else opt

# panel is n x 1 made of lists (this is necessary to make it non-rectangular)
def _find_bend(panel : npt.NDArray(dtype=list), node : npt.NDArray, model_type : str):
    if model_type == "N4B5":
        bend = _nan(3 * panel.shape[0], 4)
        cntb = 0
        for i in range(panel.shape[0]):
            p = np.array(panel[i])
            if np.size(p, axis = 0) == 4:
                l1 = np.linalg.norm(node[p[1], :] - node[p[3], :])
                l2 = np.linalg.norm(node[p[4], :] - node[p[2], :])
                lclbend = []
                if l1 > l2: lclbend = [2, 4, 1, 3]
                else: [1, 3, 2, 4]
                cntb += 1
                bend[cntb, :] = p[np.array(lclbend)]
            elif np.size(p, axis = 0) > 4:
                bi_edge = _divide_polygon()     # might need to ensure that bi_edge is 2D
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
                    bendlcl[k, :] = np.hstack(bi_edge[k], myab)
                bend[cntb + 1 : cntb + np.size(bi_edge, axis = 1)] = p[bendlcl]
                cntb += np.size(bi_edge, axis = 0)
        
        bend[np.isnan(bend[:, 1])] = []
        panelctr = []
        return (bend, node, panelctr)
    elif model_type == "N5B8":
        nn = np.size(node, axis = 0)
        bend = _nan(6 * np.size(panel), 4)
        panelctr = _nan(np.size(panel), 1)
        cntb = 0
        cntc = 0
        for i in range(np.size(panel)):
            p = np.array(panel[i])
            if np.size(p) == 4:
                cntc += 1
                L1 = np.linalg.norm(node[p[0]] - node[p[2]])
                L2 = np.linalg.norm(node[p[3]] - node[p[1]]) 
                m = np.array(node[p[2]] - node[p[0]]).T / L1
                n = np.array(node[p[2]] - node[p[0]]).T / L2
                # TODO: Revisit this calculation- it's comparable but noticeably off from the matlab calc
                coeff, *_ = np.linalg.lstsq(np.array([m, -n]).T.astype('float64'), (node[p[1]] - node[p[0]]).astype('float64'))
                if L1 < L2: ctrnode = node[p[0]] + m.T * coeff[0]
                else: ctrnode = node[p[1]] + n.T * coeff[1]
                # If 1,2,3,4 are co-planar, the two results are identical;
                # if they are not, the central node is placed such that the
                # panel is bended along the shorter diagonal.
                node[nn + cntc] = ctrnode
                panelctr[i] = nn + cntc
                for k in range(np.size(p)):
                    cntb += 1
                    ref1 = np.mod(k - 3, np.size(p))
                    ref2 = np.mod(k - 1, np.size(p))
                    bend[cntb] = [nn + cntc, p[k], p[ref1], p[ref2]]
            elif np.size(p) > 4:
                cntc += 1
                ctrnode = _find_opt_center(node[p])
                node[nn + cntc] = ctrnode      #TODO: this exceeds teh size of node, need to resize. Potentially just add a node for each panel
                panelctr[i] = nn + cntc
                for k in range(np.size(p)):
                    cntb += 1
                    ref1 = np.mod(k - 3, np.size(p))
                    ref2 = np.mod(k - 1, np.size(p))
                    bend[cntb] = [nn + cntc, p[k], p[ref1], p[ref2]]
        bend[np.isnan(bend[:, 1])] = []    
        return (bend, node, panelctr)

def _findfdbd(panel, bend):
    comp_max = np.vectorize(lambda x: max(x))
    nn = np.max(comp_max(panel))
    # triangularization
    panel_size_func = np.vectorize(len)
    panel_size = panel_size_func(panel)
    panels_3 = np.size(panel_size[panel_size == 3])
    ptri = np.empty((panels_3, 1))
    flg = np.where(panel_size == 3)
    for i in range(panels_3):
        ptri[i] = panel[flg[i]]
    trigl_raw = np.array([bend[:, [0, 2, 1]], ptri]).T  # there's no way these share dimensions
    # trigl_raw_sort = np.sort(trigl_raw, 1)
    trigl, uniqidx = np.unique(trigl_raw, axis = 0, return_index = True)
    # make connectivity matrix
    comm = np.zeros((nn, np.size(trigl, 0)))
    for i in range(np.size(trigl)):
        comm[trigl[i], i] = 1
    # find fold lines
    Ge = np.matmul(comm.T, comm)
    mf, me = np.nonzero(np.triu(Ge[Ge == 2])) # find triangular meshes w/ 2 common nodes
    fold = np.zeros(np.size(mf), 4)
    for i in range(np.size(mf)):
        # find shared vertices and the vertices of neighboring triangles
        link, ia, ib = np.intersect1d(trigl[mf[i]], trigl[me[i]], return_indices = True)
        oftpa = np.setdiff1d(np.arange(3), ia)
        oftpb = np.setdiff1d(np.arange(3), ib)

        # check ordering of nodes
        wrapverts = np.array([trigl_raw(uniqidx[mf[i]]), trigl_raw[uniqidx[mf[i]]]]).T
        wrap_n1 = wrapverts[: -1]
        wrap_2 = wrapverts[1 : ]
        isme = np.nonzero((wrapverts[:, :-1] == link[0]) and (wrapverts[:, 1:] == link[1]))
        if not np.size(isme) == 0:
            fold[i] = [link, trigl[me[i], oftpb], trigl[mf[i], [oftpa]]]
        else:
            print("WARNING: could not find correct ordering")
    
    fd_and_bd = np.sort(fold[:, :1], 1)
    only_bd = np.sort(bend[:, :1], 1)
    _, _, ibd = _intersect_2d(fd_and_bd, only_bd)
    fold[ibd] = []

    # look for boundaries
    edge = np.sort(np.block([[trigl[:, 0], trigl[:, 1]],[trigl[:, 1], trigl[:, 2]],[trigl[:, 2], trigl[:, 0]]]), 1)
    u, _, n = np.unique(edge, axis = 0, return_inverse=True)
    counts = np.bincount(n)
    bdry = u[counts==1]
    return fold, bdry, trigl


def _nan(*shape):
    arr = np.empty(*shape)
    arr.fill(np.nan)
    return arr


def _divide_polygon(poly_coord : npt.NDArray) -> npt.NDArray:
    if (poly_coord.shape[0]) <= 3:
        return []
    else:
        G = np.triu(np.ones((poly_coord.shape[0], poly_coord.shape[0])), 2)
        G[0, -1] = 0
        I, J = np.nonzero(G)
        L2 = np.sum(np.power(poly_coord[I, :] - poly_coord[J, :], 2), axis = 1)     # this seems potentially incorrect
        ind_min = np.argmin(L2, axis = 0)    # row indices of minimum in each col
        bi_edge = np.sort(np.array([I[ind_min], J[ind_min]]).T, axis = 0)
        T1 = np.concatenate((np.arange(0, bi_edge[0] + 1), np.arange(bi_edge[1], poly_coord.shape[0])))
        T2 = np.arange(bi_edge[0], bi_edge[1] + 1)
        return np.array([bi_edge,
                        T1[_divide_polygon(poly_coord[T1, :])],
                        T2[_divide_polygon(poly_coord[T2, :])]])

def _find_opt_center(poly_coord : npt.NDArray):
    G = np.triu(np.ones((np.size(poly_coord, 0), 2)))
    G[0, -1] = 0
    I, J = np.nonzero(G)
    L2 = np.sum(np.power(poly_coord[I] - poly_coord[J], 2), axis = 0)

    obj_fun = lambda xc: np.sum(np.divide(np.sqrt(np.sum(np.power(np.cross((poly_coord[J] - xc).T, (poly_coord[I] - xc).T), 2), 0)).T, np.power(L2, 1)), 0)
    idmin = np.min(L2)
    XC01 = (poly_coord[I[idmin]] + poly_coord[J[idmin]]) / 2
    XC02 = np.sum(poly_coord, 0) / np.size(poly_coord, 0)

    res1 : OptimizeResult = minimize(obj_fun, XC01, method = "BFGS")
    res2 : OptimizeResult = minimize(obj_fun, XC02, method = "BFGS")

    xc = res1.x if res1.fun[-1] <= res2.fun[-1] else res2.x
    print(f"MinObj (MidPoint = {res1.fun[-1]}, CenterPoint = {res2.fun[-1]})")

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
    D = np.array([node[ele[:, 1], 0] - node[ele[:, 0], 0], node[ele[:, 1], 1] - node[ele[:, 0], 1], node[ele[:, 1], 2] - node[ele[:, 0], 2]]).T[0]
    L = np.sqrt(np.power(D[:, 0], 2) + np.power(D[:, 1], 2) + np.power(D[:, 2], 2))
    D = np.array([np.divide(D[:, 0], L), np.divide(D[:, 1], L), np.divide(D[:, 2], L)]).T
    B = csr_matrix(
        (np.matrix.flatten(np.block([D, -D])),
        (
            np.matrix.flatten(npm.repmat(np.arange(ne).reshape(1, ne).T, 1, 6)), 
            np.matrix.flatten(np.array([3 * ele[:, 0] + 1 - 1, 3 * ele[:, 0] + 2- 1, 3 * ele[:, 0] + 3- 1, 3 * ele[:, 1] + 1- 1, 3 * ele[:, 1] + 2- 1, 3 * ele[:, 1] + 3- 1]).T)
        )),
        shape = (ne, 3 * nn))
    B = -B
    return B, L

if __name__ == "__main__":
    # testing
    node = np.array([[0, 0, 0], [0, 10, 0], [12, 15, 0], [30, 10, 0], [30, 0, 0], [18, -5, 0]]) * 2
    panel = np.arange(6)

    m = np.size(node, 0)
    supp = np.array([[0, 1, 1, 1], [1, 1, 0, 1], [3, 0, 0, 1]])
    load = np.array([4, 0, 0, -1])

    analy_input_opt : AnalyInputOpt = {}
    analy_input_opt["model_type"] = "N5B8"
    analy_input_opt["mater_calib"] = "auto"
    analy_input_opt["mod_elastic"] = 5e3
    analy_input_opt["poisson"] = 0.35
    analy_input_opt["thickness"] = 0.127
    analy_input_opt["l_scale_factor"] = 3
    analy_input_opt["load_type"] = "Force"
    analy_input_opt["initial_load_factor"] = 0.00001
    analy_input_opt["max_incr"] = 100 
    analy_input_opt["stop_criterion"] = lambda node, u, icrm: np.abs(u[5*3]) > 12

    truss, angles, analy_input_opt = prepare_data(node, panel, supp, load, analy_input_opt)
    print("Truss: " + str(truss))
    print("Angles: " + str(angles))
    print("AnalyInputOpt: " + str(analy_input_opt))