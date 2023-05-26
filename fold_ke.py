import numpy as np

def fold_ke(cood, lst):
    """Find the potential energy of a fold/bend I think"""
    rkj = (cood[lst[1], :] - cood[lst[0], :]).T
    rij = (cood[lst[2], :] - cood[lst[0], :]).T
    rkl = (cood[lst[1], :] - cood[lst[3], :]).T
    rmj = np.cross(rij, rkj)
    rnk = np.cross(rkj, rkl)
    interm = np.matmul(rnk.T, rij)
    sgn = (np.abs(interm) > 1e-8) * np.sign(interm) + (np.abs(interm) <= 1e-8)
    he = np.real(np.arccos(np.matmul(rmj.T, rnk) / (np.linalg.norm(rmj) * np.linalg.norm(rnk))))
    he = np.real(sgn * he)
    if (he < 0):
        he += 2 * np.pi
    return he