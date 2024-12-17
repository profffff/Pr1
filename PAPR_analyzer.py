import numpy as np
from ft_transform import ft_transform

def PAPR_analyzer(S_t, scenario):
    Nfft = scenario["Nfft"]

    # If input is in frequency domain
    if S_t.shape[1] < Nfft:
        S_t = ft_transform(S_t, "f2t", scenario)

    PAPR_full = []
    # Compute PAPR per symbol + aggregate all samples to array for CCDF

    P = np.abs(S_t) ** 2
    mean_P = np.mean(P)
    PAPR_full = P / mean_P
    PAPR_full = 10 * np.log10(PAPR_full.flatten())

    papr = scenario["papr_ccdf"]
    y = np.zeros(len(papr))
    for i in range(len(papr)):
        y[i] = np.sum(PAPR_full >= papr[i]) / len(PAPR_full)  # # of samples exceeding papr(i)
    
    CCDF = y
    return CCDF

