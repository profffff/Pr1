import numpy as np

def sinc_paper(X_t, A, N_used, TH, max_EVM, Nfft, scen):
    Nsym = X_t.shape[0]  # Number of OFDM symbols in the initial time-domain signal
    Ndac = scen['Ndac']  # Number of digital-to-analog converters
    Nant = scen['Ntx']  # Number of antennas (> Ndac, HBF architecture)

    N_samples = 16  # Number of points at the interval
    coef1 = 0  # Use only peaks in inversion
    coef2 = Nant / Ndac  # Use peaks and zeros in inversion

    Nzero = Nfft - N_used  # padding for IFFT calculation

    S_t_ant = np.transpose(X_t, (0, 2, 1))  # Initial signal to reduce PAPR

    # signal power and root mean square
    mean_power = np.mean(np.abs(S_t_ant) ** 2)

    TH_abs = TH * np.sqrt(mean_power)
    N_iter = len(TH)

    # generate SINC signal
    SINC_f = np.roll(np.concatenate((np.ones(N_used), np.zeros(Nzero))), -N_used // 2)
    SINC_t = np.fft.ifft(SINC_f) * np.sqrt(Nfft)
    SINC_t = SINC_t / SINC_t[0]  # Amplitude normalization to match peak amplitude

    # Generate set of sincs corresponding to different time positions
    SINC_mtx = np.zeros((Nfft, Nfft), dtype=complex)
    for j in range(Nfft):
        SINC_mtx[j, :] = np.roll(SINC_t, j)

    S_t_canc = np.zeros((Nsym, Nant, Nfft), dtype=complex)
    S_t_ant_new = np.zeros((Nsym, Nant, Nfft), dtype=complex)
    S_t_ant_canc_peak = np.zeros((Nsym, Nant, Nfft), dtype=complex)

    M_iter = 2  # Number of iterations for peaks positions search per interval

    # find peaks to reduce
    for i1 in range(Nsym):
        for i2 in range(Nant):
            S_t = S_t_ant[i1, i2, :].copy()

            # PAPR reduction algorithm
            for j in range(N_iter):
                S_t_canc_tmp = np.zeros(Nfft, dtype=complex)
                for p in range(M_iter):
                    Nintervals = Nfft // N_samples

                    # Peaks positions and amplitude per interval
                    sinc_Ampl = np.zeros(Nintervals, dtype=complex)
                    sinc_shift = np.zeros(Nintervals)
                    for k in range(Nintervals):
                        signal = S_t[k * N_samples:(k + 1) * N_samples]

                        Max_value = np.max(np.abs(signal))
                        Indx = np.argmax(np.abs(signal))

                        # Check for threshold exceeding and peak parameters set
                        if Max_value > TH_abs[j]:
                            sinc_Ampl[k] = signal[Indx] * (1 - TH_abs[j] / Max_value)
                            sinc_shift[k] = Indx + k * N_samples
                        else:
                            sinc_Ampl[k] = 0
                            sinc_shift[k] = 1

                        # Sum of sincs canceling found peaks
                        S_t_canc_tmp += sinc_Ampl[k] * np.roll(SINC_t, int(sinc_shift[k] - 1))

                        # Sum of delta functions canceling found peaks
                        S_t_ant_canc_peak[i1, i2, int(Indx + k * N_samples)] += sinc_Ampl[k]

                    S_t -= S_t_canc_tmp
                    S_t_canc[i1, i2, :] += S_t_canc_tmp

            # Modified signal after per-antenna PAPR reduction
            S_t_ant_new[i1, i2, :] = S_t

    # Now we need to check what happens in DAC domain
    S_t_dac_canc_sig = np.zeros((Nsym, Ndac, Nfft), dtype=complex)
    S_t_dac_canc_peak = np.zeros((Nsym, Ndac, Nfft), dtype=complex)
    S_t_ant_canc_peak_new = np.zeros((Nsym, Nant, Nfft), dtype=complex)

    for i1 in range(Nsym):
        for i3 in range(Nfft):
            sig = S_t_ant_canc_peak[i1, :, i3]

            # Check for found peaks
            indx = np.where(np.abs(sig) > 0)[0]

            if indx.size > 0:
                A_new = A[indx, :]
                sig_new = sig[indx]
                s1 = np.linalg.pinv(A_new) @ sig_new  # peaks only transform
                s2 = np.linalg.pinv(A) @ sig  # peaks + zeros transform
                S_t_dac_canc_peak[i1, :, i3] = coef1 * s1 + coef2 * s2
            else:
                S_t_dac_canc_peak[i1, :, i3] = np.zeros(Ndac)

        # Go back to antenna domain
        S_t_ant_canc_peak_new[i1, :, :] = A @ S_t_dac_canc_peak[i1, :, :]

        # Convolve peaks with sinc functions in DAC domain to satisfy spectrum mask
        for i2 in range(Ndac):
            s_t = S_t_dac_canc_peak[i1, i2, :]
            S_t_dac_canc_sig[i1, i2, :] = s_t @ SINC_mtx

    # Check for EVM and reduce CFR noise proportionally
    EVM_approx = np.sqrt(np.sum(np.abs(S_t_dac_canc_sig) ** 2) / np.sum(np.abs(S_t_ant) ** 2))
    S_t_dac_canc_sig /= max(EVM_approx / max_EVM, 1)

    # Generate the final signal at antennas to analyze PAPR
    S_t_ant_new2 = np.zeros((Nsym, Nant, Nfft), dtype=complex)
    for i1 in range(Nsym):
        S_t_ant_new2[i1, :, :] = A @ S_t_dac_canc_sig[i1, :, :]

    dS = S_t_ant_new2
    dX = np.transpose(dS, (0, 2, 1))
    return dX