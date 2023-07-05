from prepare_data import Truss, Angles, AnalyInputOpt
import numpy as np
import numpy.typing as npt
from typing import Tuple
from src.util.globalk_fast_ver import globalk_fast_ver

def path_analysis(truss : Truss, angles : Angles, analy_input_opt : AnalyInputOpt, do_prints : bool) -> Tuple[npt.NDArray, npt.NDArray]:
    """Performs a simulation of the origami structure with the given inputs

    Parameters
    ----------
    truss : Truss
        The truss ouput of prepare_data. The u_0 field, which defines displacements for the initial simulation state, can also be populated, and this is not done in prepare_data.
    angles : Angles
        The angles output of prepare_data.
    analy_input_opt : AnalyInputOpt
        The analy_input_opt output of prepare data
    do_prints : bool
        If true, the current increment, iteration, and error of the simulation will be printed to the console at every step. Otherwise, no outputs will be printed

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray]
        This is a tuple containing u_his and f_his.
        1. u_his - History of nodal displacements at the end of each increment, stored in a Ndof ×Nicrm array.
        2. f_his - In Force mode, this is a Nicrm × 1 array that stores the values of load factors at the end of each increment. In Displacement mode, this is a Nicrm × Ndisp array, whose columns store the negative values of the resistant forces in the degrees of freedom that are imposed with displacement loads, where Ndisp is the number of imposed degrees of freedom.

    Raises
    ------
    ValueError
        If an illegal value is passed in, this will be raised
    """
    
    tol = 1e-6
    max_iter = 50
    node = truss.get("node")
    if node is None:
        raise ValueError()

    u = truss.get("u_0")
    if not isinstance(u, np.ndarray):
        u = np.zeros((3 * np.size(node, 0), 1))
    else:
        u = u.copy()


    stop = analy_input_opt.get("stop_criterion")
    assert stop != None

    fixed_dofs = truss.get("fixed_dofs")
    assert type(fixed_dofs) == np.ndarray
    
    if (analy_input_opt.get("load_type") == "force"):
        max_icr = analy_input_opt.get("max_incr")
        assert type(max_icr) == int
        b_lambda = analy_input_opt.get("initial_load_factor")
        assert type(b_lambda) == float or type(b_lambda) == int
        u_his = np.zeros((3 * np.size(node, 0), max_icr))

        free_dofs = np.setdiff1d(np.arange(3 * np.size(node, 0)), fixed_dofs)
        lmd = 0
        icrm = -1
        mul = np.block([u, u])
        f_his = np.zeros((max_icr, 1))
        dupp1 = sinal = dupc1 = numgsp = 0
        t = analy_input_opt.get("load")
        F = None
        if isinstance(t , np.ndarray):
            F = t
        while (icrm < max_icr - 1 and (not stop(node, u, icrm))):
                icrm += 1
                iter = -1
                err = 1
                if do_prints:
                    print(f"icrm = {icrm}, lambda = {lmd:6.4f}")
                ad = analy_input_opt.get("adaptive_load")
                if ad != None:
                    F = ad(node, u, icrm)
                if F is None:
                    raise ValueError()
                while err > tol and iter < max_iter:
                    iter += 1
                    IF, K = globalk_fast_ver(u, node, truss, angles, True)
                    assert type(K) == npt.NDArray
                    R = lmd * F - IF
                    MRS = np.hstack((F, R))
                    mul[free_dofs, :] = np.linalg.solve(K.astype(float)[np.ix_(free_dofs, free_dofs)], MRS.astype(float)[free_dofs])
                    d_up = mul[:, 0]
                    d_ur = mul[:, 1]
                    if iter == 0: d_ur *= 0
                    dlmd, dupp1, sinal, dupc1, numgsp = nlsmgd(icrm, iter, d_up, d_ur, b_lambda, dupp1, sinal, dupc1, numgsp)
                    if isinstance(dlmd, np.ndarray):
                        dlmd = dlmd[0]
                    d_ut = dlmd * d_up + d_ur
                    u += d_ut.reshape((-1, 1))
                    err = np.linalg.norm(d_ut[free_dofs])
                    lmd += dlmd
                    if do_prints:
                        print(f"\titer = {iter}, err={err:6.4f}, dlambda = {dlmd:6.4f}")
                    if err > 1e8:
                        print("Divergence!")
                        break
                if iter > 14:
                    b_lambda /= 2
                    if do_prints:
                        print("Reduce constraint radius...")
                    icrm -= 1
                    u = u_his[:, [max(icrm, 0)]]
                    lmd = f_his[max(icrm, 0)][0]
                elif iter < 2:
                    if do_prints:
                        print("Increase constraint radius...")
                    b_lambda *= 1.5
                    u_his[:, [icrm]] = u
                    f_his[icrm] = lmd
                else:
                    u_his[:, [icrm]] = u
                    f_his[icrm] = lmd
    elif analy_input_opt.get("load_type") == "displacement":
        disp_step = analy_input_opt.get("disp_step")
        assert type(disp_step) == int
        u_his = np.zeros((3 * np.size(node, 0), disp_step * 2))
        ld = analy_input_opt.get("load")
        adl = analy_input_opt.get("adaptive_load")
        if ld is not None:
            fdsp = ld / disp_step
        elif adl is not None: 
            fdsp = adl(node, u, 1) # Might need to be 0
        else: 
            raise ValueError()
        imp_dofs = np.nonzero(fdsp)[0]
        free_dofs = np.setdiff1d(np.setdiff1d(range(3*np.size(node, 0)), fixed_dofs), imp_dofs)
        icrm = -1
        dspmvd = attmpts = 0
        mvstepsize = damping = 1
        f_his = np.zeros((disp_step, np.size(imp_dofs)))
        while (dspmvd <= 1 and (not stop(node, u, icrm)) and attmpts <= 20):
            icrm += 1
            if icrm == np.size(u_his, 1):
                u_his = np.hstack((u_his, np.zeros(u_his.shape)))
            if icrm == np.size(f_his, 0):
                f_his = np.vstack((f_his, np.zeros(f_his.shape)))
            iter = -1
            err = 1
            if do_prints:
                print(f"icrm = {icrm}, dspimps = {dspmvd:6.4f}")
            if adl is not None:
                fdsp = adl(node, u, icrm)
            u += mvstepsize * fdsp
            u[fixed_dofs] = 0
            while (err > tol and iter < max_iter - 1):
                iter += 1
                IF, k = globalk_fast_ver(u, node, truss, angles, True)
                assert k is not None
                du = np.zeros((3 * np.size(node, 0), 1))
                du[free_dofs] = np.linalg.solve(k[np.ix_(free_dofs, free_dofs)], -IF[free_dofs])
                err = np.linalg.norm(du[free_dofs])
                u += damping * du
                if do_prints:
                    print(f"\titer = {iter}, err = {err:6.4f}")
            
            if iter >= (((mvstepsize > 1) + 1) * max_iter / (damping + 1)) - 1:
                # an aggressive step needs more iterations
                attmpts += 1
                icrm -= 1
                if attmpts <= 10:
                    mvstepsize *= 0.5
                    if do_prints:
                        print("Taking a more conservative step...")
                else:
                    mvstepsize = max(mvstepsize, 1) * 1.5
                    damping *= 0.75
                    if do_prints:
                        print("Taking a more aggressive step...")
                u = u_his[:, [max(icrm, 0)]] # restore displacement
            else:
                dspmvd += mvstepsize / disp_step
                attmpts = 0
                damping = 1
                if mvstepsize < 1:
                    mvstepsize = min(mvstepsize * 1.1, 1)
                else:
                    mvstepsize = max(mvstepsize * 0.9, 1)
                u_his[:, [icrm]] = u
                f_end, _ = globalk_fast_ver(u, node, truss, angles, False)
                f_his[icrm, :] = -f_end[imp_dofs].T
    else:
        raise ValueError("Unknown load type!!!")
    icrm += 1
    u_his = np.delete(u_his, range(icrm, np.size(u_his, 1)), axis = 1)
    f_his = np.delete(f_his, range(icrm, np.size(f_his, 0)), axis = 0)
    return u_his, f_his




def mat_dot(A : npt.NDArray, B : npt.NDArray):
    return np.sum(A.conj()*B, axis=0)

def nlsmgd(step, ite, dup : npt.NDArray, dur, cmp, dupp1, sinal, dupc1, numgsp):
    if ite == 0:
        if step == 0:
            sinal = np.sign(mat_dot(dup, dup))
            dl = cmp
            numgsp = mat_dot(dup, dup)
        else:
            sinal *= np.sign(mat_dot(dupp1, dup))
            gsp = numgsp / mat_dot(dup, dup)
            dl = sinal * cmp * np.sqrt(gsp)
        dupp1 = dup.copy()
        dupc1 = dup.copy()
    else:
        dl = -mat_dot(dupc1, dur) / mat_dot(dupc1, dup)

    return dl, dupp1, sinal, dupc1, numgsp