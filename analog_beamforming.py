import numpy as np

def analog_beamforming(H, scenario):
    Ntx = scenario["Ntx"]
    Ndac = scenario["Ndac"]


    # Same as beamspace selection (choose powerful directions from FFT matrix)
    F = np.fft.fft(np.eye(Ntx))

    H = H.reshape(-1, Ntx)

    P = H @ F

    P = np.sum(np.abs(P)**2, axis=0)

    idx = np.argsort(P)[::-1]

    Fa = F[:, idx[:Ndac]] / np.sqrt(Ntx)

    return Fa