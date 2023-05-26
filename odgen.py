import numpy as np
from numpy.typing import NDArray

def ogden(Ex : NDArray, C0: int, include_wb : bool):
    # ogden hyperelastic constitutive model for bar elements
    alfa = np.array([5, 1])
    # alfa = [3,1]; # Linear
    a = alfa[0]
    b = alfa[1]
    pstr = np.real(np.sqrt(2 * Ex + 1))
    C0 = np.full(pstr.shape, C0)
    Ct = np.multiply(C0 / (a - b), ((a - 2) * np.power(pstr, a - 4)) - ((b - 2) * np.power(pstr, b - 4)))
    Sx = np.multiply(C0 / (a - b), np.power(pstr, a - 2) - np.power(pstr, b - 2))
    if include_wb: 
        Wb = np.multiply(C0 / (a - b), ((np.power(pstr, a) - 1) / a) - ((np.power(pstr, b) - 1) / b))
        return Sx, Ct, Wb
    else:
        return Sx, Ct