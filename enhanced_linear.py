import numpy as np
import numpy.typing as npt

def enhanced_linear(he : npt.NDArray, h0, kpi, l0, limlft, limrt, incl_espr):
    limlft = limlft / 180 * np.pi
    limrt = limrt / 180 * np.pi
    # limlft: theta_1: left partition point
    # limrht: theta_2: right partition point
    if (np.size(limlft) == 1):
        limlft *= np.ones(he.shape)
    if (np.size(limrt) == 1):
        limrt *= np.ones(he.shape)

    partl = np.divide(np.pi, limlft)
    partr = np.divide(np.pi, 2 * np.pi - limrt)

    if (np.size(kpi) == 1):
        kpi *= np.ones(he.shape)
    
    rspr = np.zeros(he.shape)
    kspr = np.ndarray.copy(rspr)

    lInd = he < limlft
    rInd = he > limrt
    mInd = np.invert(np.bitwise_or(lInd, rInd))
    rspr[lInd] = np.multiply(
                    kpi[lInd], 
                    np.real(limlft[lInd] - h0[lInd])) + (
                    np.divide(
                        np.multiply(
                                kpi[lInd], 
                                np.tan(
                                    np.multiply(
                                        partl[lInd] / 2,
                                        he[lInd] - limlft[lInd]
                                    )
                                )
                            ), 
                        partl[lInd] / 2
                    )
                )
    kspr[lInd] = np.multiply(
        kpi[lInd], 
        np.power(1 / np.cos(
            np.multiply(
                partl[lInd] / 2,
                he[lInd] - limlft[lInd]
                )
            ), 2
        )
    )
    rspr[rInd] = np.multiply(
                    kpi[rInd], 
                    np.real(limrt[rInd] - h0[rInd])) + (
                    np.divide(
                        np.multiply(
                                kpi[rInd], 
                                np.tan(
                                    np.multiply(
                                        partr[rInd] / 2,
                                        he[rInd] - limrt[rInd]
                                    )
                                )
                            ), 
                        partr[rInd] / 2
                    )
                )
    kspr[rInd] = np.multiply(
        kpi[rInd], 
        np.power(1 / np.cos(
            np.multiply(
                partr[rInd] / 2,
                he[rInd] - limrt[rInd]
                )
            ), 2
        )
    )
    rspr[mInd] = np.multiply(kpi[mInd], np.real(he[mInd] - h0[mInd]))
    kspr[mInd] = kpi[mInd]
    rspr = np.multiply(l0, rspr)
    kspr = np.multiply(l0, kspr)

    if incl_espr:
        espr = np.zeros(he.shape)
        espr[lInd] = (0.5 * 
            np.multiply(
                kpi[lInd],
                np.power(np.real(h0[lInd] - limlft[lInd]), 2)
            )) + (
                np.multiply(
                    np.multiply(
                        kpi[lInd],
                        np.real(h0[lInd] - limlft[lInd])
                    ),
                    limlft[lInd] - he[lInd]
                )
            ) - (
                np.multiply(
                    np.divide(
                        4 * kpi[lInd],
                        np.power(partl[lInd], 2)
                    ),
                    np.log(np.abs(np.abs(np.cos(np.multiply(
                        partl[lInd] / 2,
                        limlft[lInd] - he[lInd]
                    )))))
                )
            )
        espr[rInd] = (0.5 * 
            np.multiply(
                kpi[rInd],
                np.power(np.real(limrt[rInd] - h0[rInd]), 2)
            )) + (
                np.multiply(
                    np.multiply(
                        kpi[rInd],
                        np.real(limrt[rInd] - h0[rInd])
                    ),
                    he[rInd] - limrt[rInd]
                )
            ) - (
                np.multiply(
                    np.divide(
                        4 * kpi[rInd],
                        np.power(partr[rInd], 2)
                    ),
                    np.log(np.abs(np.abs(np.cos(np.multiply(
                        partr[rInd] / 2,
                        he[rInd] - limrt[rInd]
                    )))))
                )
            )
        espr[mInd] = np.multiply(
            0.5 * kpi[mInd],
            np.power(np.real(he[mInd] - h0[mInd]), 2)   
        )
        espr = np.multiply(l0, espr)
        return rspr, kspr, espr
    else:
        return rspr, kspr



    
    
