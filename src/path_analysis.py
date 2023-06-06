from prepare_data import Truss, Angles, AnalyInputOpt
import numpy as np
import numpy.typing as npt
from src.util.globalk_fast_ver import globalk_fast_ver

def path_analysis(truss : Truss, angles : Angles, analy_input_opt : AnalyInputOpt):
    
    tol = 1e-6
    max_iter = 50
    node = truss["node"]

    u = truss["u_0"]
    if not isinstance(u, np.ndarray):
        u = np.zeros((3 * np.size(node, 0), 1))
    
    if (analy_input_opt["load_type"] == "force"):
        max_icr = analy_input_opt["max_incr"]
        b_lambda = analy_input_opt["initial_load_factor"]
        u_his = np.zeros((3 * np.size(node, 0), max_icr))
        free_dofs = np.setdiff1d(np.arange(3 * np.size(node, 0)), truss["fixed_dofs"])
        lmd = 0
        icrm = -1
        mul = np.block([u, u])
        f_his = np.zeros((max_icr, 1))
        dupp1 = sinal = dupc1 = numgsp = 0
        if isinstance(analy_input_opt["load"] , np.ndarray):
            F = analy_input_opt["load"]
        while (icrm < max_icr - 1 and (not analy_input_opt["stop_criterion"](node, u, icrm))):
                icrm += 1
                iter = -1
                err = 1
                print(f"icrm = {icrm}, lambda = {lmd:6.4f}")
                if analy_input_opt.get("adaptive_load") != None:
                    F = analy_input_opt["adaptive_load"](node, u, icrm)
                while err > tol and iter < max_iter:
                    iter += 1
                    IF, K = globalk_fast_ver(u, node, truss, angles, True)
                    R = lmd * F - IF
                    MRS = np.hstack((F, R))
                    mul[free_dofs, :] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)].toarray(), MRS[free_dofs])
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
                    print(f"\titer = {iter}, err={err:6.4f}, dlambda = {dlmd:6.4f}")
                    if err > 1e8:
                        print("Divergence!")
                        break
                if iter > 14:
                    b_lambda /= 2
                    print("Reduce constraint radius...")
                    icrm -= 1
                    u = u_his[:, [max(icrm, 0)]]
                    lmd = f_his[max(icrm, 0)][0]
                elif iter < 2:
                    print("Increase constraint radius...")
                    b_lambda *= 1.5
                    u_his[:, [icrm]] = u
                    f_his[icrm] = lmd
                else:
                    u_his[:, [icrm]] = u
                    f_his[icrm] = lmd
    elif analy_input_opt["load_type"] == "displacement":
        u_his = np.zeros((3 * np.size(node, 0), analy_input_opt["disp_step"] * 2))
        if analy_input_opt.get("load") is not None:
            fdsp = analy_input_opt["load"] / analy_input_opt["disp_step"]
        else: 
            fdsp = analy_input_opt["adaptive_load"](node, u, 1) # Might need to be 0
        imp_dofs = np.nonzero(fdsp)
        free_dofs = np.setdiff1d(np.setdiff1d(range(3*np.size(node, 0)), truss["fixed_dofs"]), imp_dofs)
        icrm = -1
        dspmvd = attmpts = 0
        mvstepsize = damping = 1
        f_his = np.zeros((analy_input_opt["disp_step"], np.size(imp_dofs)))
        while (dspmvd <= 1 and (not analy_input_opt["stop_criterion"](node, u, icrm)) and attmpts <= 20):
            icrm += 1
            iter = -1
            err = 1
            print(f"icrm = {icrm}, dspimps = {dspmvd:6.4f}")
            if analy_input_opt.get("adaptive_load") is not None:
                fdsp = analy_input_opt["adaptive_load"](node, u, icrm)
            u += mvstepsize * fdsp
            u[truss["fixed_dofs"]] = 0
            while (err > tol and iter < max_iter):
                iter += 1
                IF, k = globalk_fast_ver(u, node, truss, angles, True)
                du = np.zeros((3 * np.size(node, 0), 1))
                du[free_dofs] = np.linalg.solve(k[free_dofs, free_dofs], -IF[free_dofs])
                err = np.linalg.norm(du[free_dofs])
                u += damping * du
                print(f"\titer = {iter}, err = {err:6.4f}")
            
            if iter >= (((mvstepsize > 1) + 1) * max_iter / (damping + 1)):
                # an aggressive step needs more iterations
                attmpts += 1
                icrm -= 1
                if attmpts <= 10:
                    mvstepsize *= 0.5
                    print("Taking a more conservative step...")
                else:
                    mvstepsize = max(mvstepsize, 1) * 1.5
                    damping *= 0.75
                    print("Taking a more aggressive step...")
                u = u_his[:, [max(icrm, 1)]] # restore displacement
            else:
                dspmvd += mvstepsize / analy_input_opt["disp_step"]
                attmpts = 0
                damping = 1
                if mvstepsize < 1:
                    mvstepsize = min(mvstepsize * 1.1, 1)
                else:
                    mvstepsize = max(mvstepsize * 0.9, 1)
                u_his[:, icrm] = u
                f_end = globalk_fast_ver(u, node, truss, angles, False)
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