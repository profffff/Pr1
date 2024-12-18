import numpy as np
def find_intervals(x_t):
    if len(x_t.shape) == 1:  # 1D input
        Nfft = len(x_t)
        abs_signal = abs(x_t)
        min_inds = []
        for k in range(1, Nfft - 1):
            if abs_signal[k] < abs_signal[k - 1] and abs_signal[k] < abs_signal[k + 1]:
                min_inds.append(k)
        if not min_inds or min_inds[0] != 0:
            min_inds.insert(0, 0)
        if min_inds[-1] != Nfft - 1:
            min_inds.append(Nfft - 1)
        return min_inds
    elif len(x_t.shape) == 3:  # 3D input
        Nsym, Nfft, Nant = x_t.shape
        intervals = []
        for i1 in range(Nsym):
            for i2 in range(Nant):
                signal = x_t[i1, :, i2]
                abs_signal = abs(signal)
                min_inds = []
                for k in range(1, Nfft - 1):
                    if abs_signal[k] < abs_signal[k - 1] and abs_signal[k] < abs_signal[k + 1]:
                        min_inds.append(k)
                if not min_inds or min_inds[0] != 0:
                    min_inds.insert(0, 0)
                if min_inds[-1] != Nfft - 1:
                    min_inds.append(Nfft - 1)
                intervals.append(min_inds)
        return intervals
    else:
        raise ValueError(f"Unexpected shape for `x_t`: {x_t.shape}")

