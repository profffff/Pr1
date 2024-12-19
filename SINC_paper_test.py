import numpy as np

def sinc_paper_test(X_t, A, N_used, TH, max_EVM, Nfft, scen):
    Nsym = X_t.shape[0]  # Number of OFDM symbols
    Ndac = scen['Ndac']  # Number of digital DACs
    Nant = scen['Ntx']  # Number of antennas (> Ndac, HBF architecture)

    Nsc = N_used  # Number of subcarriers (192)
    Nsamples = 16  # Grid size for sinc
    Nzero = Nfft - N_used  # Padding for IFFT

    # Ensure dimensions are consistent
    if X_t.shape[2] != Nant:
        raise ValueError(f"X_t has incompatible number of antennas: {X_t.shape[2]}, expected {Nant}.")
    if X_t.shape[1] != Nfft:
        raise ValueError(f"X_t has incompatible number of subcarriers: {X_t.shape[1]}, expected {Nfft}.")

    # Retain only the first N_used (192) subcarriers
    X_t = X_t[:, :Nsc, :]  # Slice only the first Nsc subcarriers

    # Transform signal into a global plane
    S_t_global = np.reshape(X_t, (Nsym, Nant, Nsc))

    # Define sinc kernel
    def adaptive_sinc_2d(center_x, center_y, grid_size, shape):
        x = np.arange(shape[0])
        y = np.arange(shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        X = X - center_x
        Y = Y - center_y
        radius = np.sqrt(X**2 + Y**2)
        sinc_kernel = np.sinc(radius / grid_size)
        return sinc_kernel

    # Process signal
    for i1 in range(Nsym):
        S_t_plane = S_t_global[i1, :, :]  # Get the 2D signal for the current symbol

        # Temporary signal for suppression
        S_t_canc_tmp = np.zeros_like(S_t_plane, dtype=complex)

        # Check amplitudes and suppress peaks
        for k in range(S_t_plane.shape[0]):  # Iterate over antennas
            for l in range(S_t_plane.shape[1]):  # Iterate over subcarriers
                value = np.abs(S_t_plane[k, l])
                if value > TH:
                    # Generate sinc kernel centered at (k, l)
                    sinc_kernel = adaptive_sinc_2d(k, l, Nsamples, S_t_plane.shape)

                    # Scale kernel to match the magnitude of the peak
                    sinc_ampl = S_t_plane[k, l] * (1 - TH / value)
                    S_t_canc_tmp += sinc_ampl * sinc_kernel

        # Update signal by subtracting suppression signal
        S_t_plane -= S_t_canc_tmp

        # Debug: Check suppression
        if i1 == 0:
            print(f"Peak before suppression: {np.max(np.abs(S_t_global[i1, :, :]))}")
            print(f"Peak after suppression: {np.max(np.abs(S_t_plane))}")

        S_t_global[i1, :, :] = S_t_plane

    # Restore signal to antenna domain
    S_t_ant_new = S_t_global

    # Transform back to DAC domain
    S_t_dac = np.zeros((Nsym, Ndac, Nfft), dtype=complex)
    for i1 in range(Nsym):
        temp_result = np.linalg.pinv(A) @ S_t_ant_new[i1, :, :]  # Transform to DAC domain
        temp_result_padded = np.pad(temp_result, ((0, 0), (0, Nzero)), mode='constant')  # Pad to get Nfft
        S_t_dac[i1, :, :] = temp_result_padded

    # Correct EVM
    EVM_approx = np.sqrt(np.sum(np.abs(S_t_dac) ** 2) / np.sum(np.abs(S_t_ant_new) ** 2))
    S_t_dac /= max(EVM_approx / max_EVM, 1)

    # Transform back to antenna domain
    S_t_ant_new2 = np.zeros((Nsym, Nant, Nfft), dtype=complex)
    for i1 in range(Nsym):
        S_t_ant_new2[i1, :, :] = A @ S_t_dac[i1, :, :]

    # Convert back to original format
    dX = np.transpose(S_t_ant_new2, (0, 2, 1))
    return dX
