import numpy as np
from scipy import special


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return phi, rho


def calculate_FB_bases(L1):
    maxK = (2 * L1 + 1)**2 - 1

    L = L1 + 1
    R = L1 + 0.5
    truncate_freq_factor = 1.5

    if L1 < 2:
        truncate_freq_factor = 2

    xx, yy = np.meshgrid(range(-L, L + 1), range(-L, L + 1))

    xx = xx / R
    yy = yy / R

    ugrid = np.concatenate([yy.reshape(-1, 1), xx.reshape(-1, 1)], 1)
    tgrid, rgrid = cart2pol(ugrid[:, 0], ugrid[:, 1])

    num_grid_points = ugrid.shape[0]

    kmax = 15

    bessel = np.load('bessel.npy')

    B = bessel[(bessel[:, 0] <= kmax) & (bessel[:, 3] <= np.pi * R * truncate_freq_factor)]

    idxB = np.argsort(B[:, 2])

    ang_freqs = B[idxB, 0]
    rad_freqs = B[idxB, 1]
    R_ns = B[idxB, 2]

    num_kq_all = len(ang_freqs)

    Phi_ns = np.zeros((num_grid_points, num_kq_all), np.float32)

    Psi = []
    kq_Psi = []
    num_bases = 0

    for i in range(B.shape[0]):
        ki = ang_freqs[i]
        qi = rad_freqs[i]
        rkqi = R_ns[i]

        r0grid = rgrid * R_ns[i]

        F = special.jv(ki, r0grid)
        Phi = 1. / np.abs(special.jv(ki + 1, R_ns[i])) * F
        Phi[rgrid >= 1] = 0
        Phi_ns[:, i] = Phi

        if ki == 0:
            Psi.append(Phi)
            kq_Psi.append([ki, qi, rkqi])
            num_bases = num_bases + 1
        else:
            Psi.append(Phi * np.cos(ki * tgrid) * np.sqrt(2))
            Psi.append(Phi * np.sin(ki * tgrid) * np.sqrt(2))
            kq_Psi.append([ki, qi, rkqi])
            kq_Psi.append([ki, qi, rkqi])
            num_bases = num_bases + 2

    Psi = np.array(Psi)
    kq_Psi = np.array(kq_Psi)

    num_bases = Psi.shape[1]

    if num_bases > maxK:
        Psi = Psi[:maxK]
        kq_Psi = kq_Psi[:maxK]
    num_bases = Psi.shape[0]
    p = Psi.reshape(num_bases, 2 * L + 1, 2 * L + 1).transpose(1, 2, 0)
    psi = p[1:-1, 1:-1, :]
    psi = psi.reshape((2 * L1 + 1)**2, num_bases)
    c = np.sqrt(np.sum(psi**2, 0).mean())
    psi = psi / c

    return psi, c, kq_Psi
