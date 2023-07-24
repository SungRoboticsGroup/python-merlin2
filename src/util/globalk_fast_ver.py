import numpy as np
from .dicts import Truss, Angles
import numpy.typing as npt
from typing import Tuple, Optional
import numpy.matlib as npm

def _icross(a : npt.NDArray, b : npt.NDArray, cast_to_3d = False) -> npt.NDArray:
    if len(a.shape) == 3:
        a = a[:, :, 0]
    if len(b.shape) == 3:
        b = b[:, :, 0]
    a = np.vstack((
        a[1] * b[2] - a[2] * b[1], 
        a[2] * b[0] - a[0] * b[2], 
        a[0] * b[1] - a[1] * b[0], 
        ))
    return a if not cast_to_3d else np.expand_dims(a, 2)

def globalk_fast_ver(ui : npt.NDArray, node : npt.NDArray, truss : Truss, angles : Angles, incl_k : bool) -> Tuple[npt.NDArray, Optional[npt.NDArray]]:
    nn = np.size(node, 0)
    nodenw = np.zeros((nn, 3))
    nodenw[:, [0]] = node[:, [0]] + ui[ : : 3]
    nodenw[:, [1]] = node[:, [1]] + ui[1 : : 3]
    nodenw[:, [2]] = node[:, [2]] + ui[2 : : 3]

    e_dof_b = np.kron(truss.get("bars"), np.full((1, 3), 3)) + np.tile(np.array([0, 1, 2]), (np.size(truss["bars"], 0), 2))
    du = (ui[e_dof_b[:, : 3]] - ui[e_dof_b[:, 3 : 6]])[:, :, 0]
    b = truss.get("b")
    assert type(b) == np.ndarray
    ex = (b @ ui) / truss.get("l") + 0.5 * (np.sum(np.power(du, 2), 1, keepdims=True) / np.power(truss["l"].reshape((-1, 1)), 2))
    cm = truss.get("cm")
    assert cm != None
    sx, et, _ = cm(ex, False)
    duelem = np.block([du, -du])
    Du = np.zeros((np.size(et), np.size(ui)))
    Du[np.tile(np.arange(np.size(et)), (1, 6)).reshape((-1,)), e_dof_b.reshape((-1,), order="F")] = duelem.reshape((-1,), order="F")
    fx = sx * truss.get("a")
    IFb = (np.sum(truss.get("b").T.dot(fx), 1, keepdims=True) + np.sum(Du.T.dot(fx / truss["l"]), 1, keepdims=True))
    tmp = np.zeros((np.size(et), np.size(et)))
    tmp[np.arange(np.size(et)), np.arange(np.size(et))] = ((et * truss.get("a")) / truss.get("l")).flatten()
    kel = truss.get("b").T @ tmp @ truss.get("b")
                    
    t = np.zeros((np.size(et), np.size(et)))
    t[np.arange(np.size(et)), np.arange(np.size(et))] = ((et * truss["a"]) / np.square(truss["l"])).flatten()
    k1 = Du.T @ t @ truss["b"] + truss["b"].T @ t @ Du

    u = np.zeros((np.size(et), np.size(et)))
    u[np.arange(np.size(et)), np.arange(np.size(et))] = ((et * truss["a"]) / np.power(truss["l"], 3)).flatten()
    k2 = Du.T @ u @ Du
    
    G = np.zeros((np.size(et), nn))
    G[np.hstack([np.arange(np.size(et))] * 2), truss["bars"].reshape((-1,), order="F")] = np.vstack((np.ones((1, np.size(et))), -np.ones((1, np.size(et))))).flatten()
    
    tmp = G.T @ (G * (fx / truss["l"]))
    ia, ja = np.nonzero(tmp)
    sa = np.asarray(tmp[tmp != 0])
    ik = 3 * ia.reshape((-1, 1)) + np.arange(3)
    jk = 3 * ja.reshape((-1, 1)) + np.arange(3)
    kg = np.zeros((3*nn, 3*nn))
    kg[ik.reshape((-1,), order="F"), jk.reshape((-1,), order="F")] = np.tile(sa, (1, 3)).flatten()
    kg = 0.5 * (kg + kg.T)
    Kb = (kel + k1 + k2) + kg

    rotspr = np.vstack((angles["bend"], angles["fold"])).astype(int)
    e_dof_d = (np.kron(rotspr, np.full((1, 3), 3)) + npm.repmat([0, 1, 2], np.size(rotspr, 0), 4)).T
    rkj = (nodenw[rotspr[:, 1]] - nodenw[rotspr[:, 0]]).T
    rij = (nodenw[rotspr[:, 2]] - nodenw[rotspr[:, 0]]).T
    rkl = (nodenw[rotspr[:, 1]] - nodenw[rotspr[:, 3]]).T
    rmj = _icross(rij, rkj)
    rnk = _icross(rkj, rkl)

    dt_rnkrij = np.sum(rnk * rij, 0)
    sgn = (np.abs(dt_rnkrij) > 1e-8) * np.sign(dt_rnkrij) + (np.abs(dt_rnkrij) <= 1e-8)
    dt_rmjrnk = np.sum(rmj * rnk, 0)
    rmj2 = np.sum(np.square(rmj), 0)
    norm_rmj = np.sqrt(rmj2)
    rkj2 = np.sum(np.square(rkj), 0)
    norm_rkj = np.sqrt(rkj2)
    rnk2 = np.sum(np.square(rnk), 0)
    norm_rnk = np.sqrt(rnk2)

    he = sgn * np.real(np.arccos(np.clip(dt_rmjrnk / (norm_rmj * norm_rnk), -1, 1)))
    he[he < 0] = 2 * np.pi + he[he < 0]
    rspr = np.zeros(np.size(he)).reshape((-1, 1))
    kspr = np.ndarray.copy(rspr)
    rspr_b, kspr_b, _ = angles["cm_bend"](he[: np.size(angles["bend"], 0)].reshape((-1, 1)), angles["pb_0"], angles["k_b"], truss["l"][: np.size(angles["bend"], 0)], False)
    rspr_f, kspr_f, _ = angles["cm_fold"](he[np.size(angles["bend"], 0) : ].reshape((-1, 1)), angles["pf_0"], angles["k_f"].reshape((-1, 1)), truss["l"][np.size(angles["bend"], 0) : np.size(angles["bend"], 0) + np.size(angles["fold"], 0)], False)
    rspr[: np.size(angles["bend"], 0)] = rspr_b
    kspr[: np.size(angles["bend"], 0)] = kspr_b
    rspr[np.size(angles["bend"], 0) :] = rspr_f.reshape((-1,1))
    kspr[np.size(angles["bend"], 0) :] = kspr_f.reshape((-1,1))
    dt_rijrkj = np.sum(rij * rkj, 0)
    dt_rklrkj = np.sum(rkl * rkj, 0)

    di = rmj * (norm_rkj / rmj2)
    dl = -rnk * (norm_rkj / rnk2)
    dj = di * ((dt_rijrkj / rkj2) - 1) - dl * (dt_rklrkj / rkj2)
    dk = -di * (dt_rijrkj / rkj2) + dl * ((dt_rklrkj / rkj2) - 1)
    jhe_dense = np.vstack([dj, dk, di, dl])
    jhe = np.zeros((np.size(ui), np.size(he)))
    jhe[e_dof_d.reshape((-1,), order="F"), np.repeat(np.arange(np.size(he)), 12)] = jhe_dense.reshape((-1,), order="F")
    ifbf = np.sum(jhe * rspr.T, 1, keepdims=True)
    IF = IFb + ifbf

    if incl_k:
        rkj3 = np.expand_dims(rkj, 2)
        rij3 = np.expand_dims(rij, 2)
        rkl3 = np.expand_dims(rkl, 2)
        rmj3 = np.expand_dims(rmj, 2)
        rnk3 = np.expand_dims(rnk, 2)

        dt_rnkrij3 = np.sum(rnk3 * rij3, 0, keepdims=True)
        sgn3 = (np.abs(dt_rnkrij3) > 1e-8) * np.sign(dt_rnkrij3) + np.abs(dt_rnkrij3 <= 1e-8)
        dt_rmjrnk3 = np.sum(rmj3 * rnk3, 0, keepdims=True)
        rmj23 = np.sum(np.square(rmj3), 0, keepdims=True)
        norm_rmj = np.sqrt(rmj23)
        rkj23 = np.sum(np.square(rkj3), 0, keepdims=True)
        norm_rkj3 = np.sqrt(rkj23)
        rnk23 = np.sum(np.square(rnk3), 0, keepdims=True)
        norm_rnk3 = np.sqrt(rnk23)

        dt_rijrkj3 = np.sum(rij3 * rkj3, 0)
        dt_rklrkj3 = np.sum(rkl3 * rkj3, 0)
        
        dii = -1 * ((np.transpose(rmj3, (0, 2, 1)) * np.transpose(_icross(rkj3, rmj3, True), (2, 0, 1)) + 
               np.transpose(np.transpose(rmj3, (0, 2, 1)) * np.transpose(_icross(rkj3, rmj3, True), (2, 0, 1)), (1, 0, 2)))
             * np.transpose(norm_rkj3 / (np.square(rmj23)), (2, 0, 1)))
        
        dij = (-(np.transpose(rmj3, (0, 2, 1)) * np.transpose(rkj3, (2, 0, 1))) * np.transpose(1 / (rmj23 * norm_rkj3), (2, 0, 1)) + 
            ((np.transpose(rmj3, (0, 2, 1)) * np.transpose(_icross(rkj3 - rij3, rmj3, True), (2, 0, 1))
             + np.transpose(np.transpose(rmj3, (0, 2, 1)) * np.transpose(_icross(rkj3-rij3, rmj3, True), (2, 0, 1)), (1, 0, 2))) * np.transpose(norm_rkj3 / np.square(rmj23), (2, 0, 1)))
            )
        
        dik = ((np.transpose(rmj3, (0, 2, 1)) * np.transpose(rkj3, (2, 0, 1))) * np.transpose(1 / (rmj23 * norm_rkj3), (2, 0, 1)) + 
            ((np.transpose(rmj3, (0, 2, 1)) * np.transpose(_icross(rij3, rmj3, True), (2, 0, 1))
             + np.transpose(np.transpose(rmj3, (0, 2, 1)) * np.transpose(_icross(rij3, rmj3, True), (2, 0, 1)), (1, 0, 2))) * np.transpose(norm_rkj3 / np.square(rmj23), (2, 0, 1)))
            )
        
        dll = (np.transpose(rnk3, (0, 2, 1)) * np.transpose(_icross(rkj3, rnk3, True), (2, 0, 1)) + 
        np.transpose(np.transpose(rnk3, (0, 2, 1)) * np.transpose(_icross(rkj3, rnk3, True), (2, 0, 1)), (1, 0, 2))) * np.transpose(norm_rkj3 / np.square(rnk23), (2, 0, 1))

        dlk = (-np.transpose(rnk3, (0, 2, 1)) * np.transpose(rkj3, (2, 0, 1)) * np.transpose(1 / (rnk23 * norm_rkj3), (2, 0, 1)) - 
        (np.transpose(rnk3, (0, 2, 1)) * np.transpose(_icross(rkj3 - rkl3, rnk3, True), (2, 0, 1)) + 
        np.transpose(np.transpose(rnk3, (0, 2, 1)) * np.transpose(_icross(rkj3 - rkl3, rnk3, True), (2, 0, 1)), (1, 0, 2))
        ) * np.transpose(norm_rkj3 / np.square(rnk23), (2, 0, 1)))

        dlj = (np.transpose(rnk3, (0, 2, 1)) * np.transpose(rkj3, (2, 0, 1)) * np.transpose(1 / (rnk23 * norm_rkj3), (2, 0, 1)) - 
               (np.transpose(rnk3, (0, 2, 1)) * np.transpose(_icross(rkl3, rnk3, True), (2, 0, 1)) + 
                np.transpose(np.transpose(rnk3, (0, 2, 1)) * np.transpose(_icross(rkl3, rnk3, True), (2, 0, 1)), (1, 0, 2))) * 
                np.transpose(norm_rkj3 / np.square(rnk23), (2, 0, 1)))
        
        dT1jj = ((rkj * (-1 + 2 * dt_rijrkj / rkj2)) - rij) * (1 / rkj2)
        dT2jj = ((rkj * (2 * dt_rklrkj / rkj2)) - rkl) * (1 / rkj2)
        djj = (np.transpose(np.expand_dims(di, 2), (0, 2, 1)) * np.transpose(np.expand_dims(dT1jj, 2), (2, 0, 1)) + 
               dij * np.transpose((dt_rijrkj3 / rkj23) - 1, (2, 0, 1)) - 
               np.transpose(np.expand_dims(dl, 2), (0, 2, 1)) * np.transpose(np.expand_dims(dT2jj, 2), (2, 0, 1)) -
               dlj * np.transpose( dt_rklrkj3 / rkj23, (2, 0, 1))
               )
        
        dT1jk = (rkj * (-2 * dt_rijrkj / rkj2) + rij) / rkj2
        dT2jk = (rkj * (1 - 2*dt_rklrkj / rkj2) + rkl) / rkj2
        djk = (np.transpose(np.expand_dims(di, 2), (0, 2, 1)) * np.transpose(np.expand_dims(dT1jk, 2), (2, 0, 1)) + 
                dik * np.transpose((dt_rijrkj3 / rkj23) - 1, (2, 0, 1)) -
                np.transpose(np.expand_dims(dl, 2), (0, 2, 1)) * np.transpose(np.expand_dims(dT2jk, 2), (2, 0, 1)) -
                dlk * np.transpose(dt_rklrkj3 / rkj23, (2, 0, 1))
                )
        
        dT1kk = dT2jk
        dT2kk = dT1jk
        dkk = (np.transpose(np.expand_dims(dl, 2), (0, 2, 1)) * np.transpose(np.expand_dims(dT1kk, 2), (2, 0, 1)) +
               dlk * np.transpose(dt_rklrkj3 / rkj23 - 1, (2, 0, 1)) -
               np.transpose(np.expand_dims(di, 2), (0, 2, 1)) * np.transpose(np.expand_dims(dT2kk, 2), (2, 0, 1)) -
               dik * np.transpose(dt_rijrkj3 / rkj23, (2, 0, 1)))
        
        hp = np.zeros((12, 12, np.size(he)))
        hp[:3, :3, :] = djj
        hp[6:9, 6:9, :] = dii
        hp[3:6, 3:6, :] = dkk
        hp[9:12, 9:12, :] = dll
        hp[:3, 3:6, :] = djk
        hp[3:6, :3, :] = np.transpose(djk, (1, 0, 2))
        hp[6:9, :3, :] = dij
        hp[:3, 6:9, :] = np.transpose(dij, (1, 0, 2))
        hp[9:12, :3, :] = dlj
        hp[:3, 9:12, :] = np.transpose(dlj, (1, 0, 2))
        hp[6:9, 3:6, :] = dik
        hp[3:6, 6:9, :] = np.transpose(dik, (1, 0, 2))
        hp[9:12, 3:6, :] = dlk
        hp[3:6, 9:12, :] = np.transpose(dlk, (1, 0, 2))

        khe_dense = (np.transpose(np.expand_dims(jhe_dense, 2), (0, 2, 1)) * np.transpose(np.expand_dims(jhe_dense, 2), (2, 0, 1)) * np.transpose(np.expand_dims(kspr.reshape((1, -1)), 2), (2, 0, 1)) + 
                     hp * np.transpose(np.expand_dims(rspr.reshape((1, -1)), 2), (2, 0, 1)))
        
        dof_ind1 = np.transpose(np.tile(np.expand_dims(e_dof_d, 2), (1, 1, 12)), (0, 2, 1))
        dof_ind2 = np.transpose(dof_ind1, (1, 0, 2))
        kbf = np.zeros((3*nn, 3*nn))
        np.add.at(kbf, (dof_ind1.flatten(order="F"), dof_ind2.flatten(order="F")), khe_dense.flatten(order="F"))
        k = Kb + kbf
        k = (k + k.T) / 2
        tmp = np.zeros((3*nn, 3*nn))
        tmp[np.arange(3*nn), np.arange(3*nn)] = np.full((3*nn,), 1e-8)
        k = k + tmp
        return IF, k
    else:
        return IF, None
        
