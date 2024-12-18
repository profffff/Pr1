import numpy as np
from scipy.signal import butter
from scipy.fft import ifft
from find_intervals import find_intervals

def sinc_paper_updated(X_t, A, N_used, TH, max_EVM, Nfft, scen):
    # Parameters and constants
    Nsym = X_t.shape[0]  # Number of OFDM symbols in the initial time-domain signal
    Ndac = scen['Ndac']  # Number of digital-to-analog converters
    Nant = scen['Ntx']  # Number of antennas (> Ndac, HBF architecture)
    N_samples = 16  # Number of points at the interval
    coef1 = 0  # Use only peaks in inversion
    coef2 = Nant / Ndac  # Default ratio for no power loss
    Nzero = Nfft - N_used  # Padding for IFFT calculation

    # Initial signal to reduce PAPR
    S_t_ant = np.transpose(X_t, (0, 2, 1))  
    mean_power = np.mean(np.abs(S_t_ant) ** 2)
    TH_abs = TH * np.sqrt(mean_power)
    N_iter = len(TH)

    # Butterworth filter
    n = 4  # Filter order
    Fc = 0.2  # Cut-off frequency (normalized from 0 to 1)
    b, a = butter(n, Fc, btype='low')

    # Generate SINC matrix
    SINC_f = np.roll(np.concatenate([np.ones(N_used), np.zeros(Nzero)]), -N_used // 2)
    SINC_t = ifft(SINC_f) * np.sqrt(Nfft)
    SINC_t /= SINC_t[0]
    SINC_mtx = np.array([np.roll(SINC_t, j - 1) for j in range(Nfft)])

    # Initialize arrays
    S_t_canc = np.zeros_like(S_t_ant, dtype=complex)
    S_t_ant_new = np.zeros_like(S_t_ant, dtype=complex)
    S_t_ant_canc_peak = np.zeros_like(S_t_ant, dtype=complex)

    for i1 in range(Nsym):
        for i2 in range(Nant):
            S_t = S_t_ant[i1, i2, :].copy()
            min_inds = find_intervals(S_t)

            for j in range(N_iter):
                S_t_canc_tmp = np.zeros(Nfft, dtype=complex)

                for k in range(len(min_inds) - 1):
                    interval_start = min_inds[k]
                    interval_end = min_inds[k + 1] - 1
                    signal = S_t[interval_start:interval_end + 1]
                    max_value = np.max(np.abs(signal))

                    if max_value > TH_abs[j]:
                        sinc_ampl = signal[np.argmax(np.abs(signal))] * (1 - TH_abs[j] / max_value)
                        sinc_shift = interval_start + np.argmax(np.abs(signal))
                        filtered_signal = np.roll(SINC_t, sinc_shift - 1) * sinc_ampl
                        S_t_canc_tmp += filtered_signal
                        S_t_ant_canc_peak[i1, i2, sinc_shift] += sinc_ampl

                S_t -= S_t_canc_tmp
                S_t_canc[i1, i2, :] += S_t_canc_tmp
            S_t_ant_new[i1, i2, :] = S_t

    S_t_dac_canc_sig = np.zeros((Nsym, Ndac, Nfft), dtype=complex)
    S_t_dac_canc_peak = np.zeros((Nsym, Ndac, Nfft), dtype=complex)

    for i1 in range(Nsym):
        for i3 in range(Nfft):
            sig = S_t_ant_canc_peak[i1, :, i3]
            indx = np.nonzero(np.abs(sig))[0]
            if indx.size > 0:
                A_new = A[indx, :]
                sig_new = sig[indx]
                s1 = np.linalg.pinv(A_new) @ sig_new
                s2 = np.linalg.pinv(A) @ sig
                S_t_dac_canc_peak[i1, :, i3] = coef1 * s1 + coef2 * s2
            else:
                S_t_dac_canc_peak[i1, :, i3] = np.zeros(Ndac)

        for i2 in range(Ndac):
            s_t = S_t_dac_canc_peak[i1, i2, :]
            S_t_dac_canc_sig[i1, i2, :] = s_t @ SINC_mtx

    evm_approx = np.sqrt(np.sum(np.abs(S_t_dac_canc_sig) ** 2) / np.sum(np.abs(S_t_ant) ** 2))
    S_t_dac_canc_sig /= max(evm_approx / max_EVM, 1)

    S_t_ant_new2 = np.zeros((Nsym, Nant, Nfft), dtype=complex)
    for i1 in range(Nsym):
        S_t_ant_new2[i1, :, :] = A @ S_t_dac_canc_sig[i1, :, :]

    dX = np.transpose(S_t_ant_new2, (0, 2, 1))
    return dX
