import numpy as np

def super_linear_bend(he, h0, kp, l0, include_espr):
    rspr = np.sign(np.round(he - h0, 3)) * np.power(np.abs(he - h0), (4/3)) * kp * l0
    kspr = np.maximum((4 / 3) * np.power(np.abs(he - h0), (1 / 3)) * kp, 0.001 * kp) * l0
    if include_espr:
        espr = np.multiply((3/7) * np.power(np.abs(he - h0), (7 / 3)), np.multiply(kp, l0))
        return rspr, kspr, espr
    else:
        return rspr, kspr