import numpy as np

def sinc_paper(X_t, A, N_used, TH, max_EVM, Nfft, scen):
    Nsym = X_t.shape[0]  # Количество OFDM символов
    Ndac = scen['Ndac']  # Количество ЦАП
    Nant = scen['Ntx']  # Количество антенн

    N_samples = 16  # Количество отсчетов в интервале
    coef1 = 0  # Использовать только пики в инверсии
    coef2 = Nant / Ndac  # Использовать пики и нули в инверсии

    Nzero = Nfft - N_used  # Заполнение нулями для IFFT

    S_t_ant = np.transpose(X_t, (0, 2, 1))  # Преобразуем сигнал для уменьшения PAPR

    # Мощность сигнала и среднеквадратичное значение
    mean_power = np.mean(np.abs(S_t_ant) ** 2)

    TH_abs = TH * np.sqrt(mean_power)
    N_iter = len(TH)

    # Генерация SINC сигнала
    SINC_f = np.roll(np.concatenate((np.ones(N_used), np.zeros(Nzero))), -N_used // 2)
    SINC_t = np.fft.ifft(SINC_f) * np.sqrt(Nfft)
    SINC_t = SINC_t / SINC_t[0]  # Нормализация амплитуды

    # Генерация двумерной матрицы SINC для всех временных положений
    SINC_mtx = np.zeros((Nfft, Nfft), dtype=complex)
    for j in range(Nfft):
        SINC_mtx[j, :] = np.roll(SINC_t, j)

    S_t_canc = np.zeros((Nsym, Nant, Nfft), dtype=complex)
    S_t_ant_new = np.zeros((Nsym, Nant, Nfft), dtype=complex)
    S_t_ant_canc_peak = np.zeros((Nsym, Nant, Nfft), dtype=complex)

    M_iter = 2  # Количество итераций для поиска пиков в интервале

    # Поиск пиков для уменьшения
    for i1 in range(Nsym):  # Для каждого символа
        for i2 in range(Nant):  # Для каждой антенны
            S_t = S_t_ant[i1, i2, :].copy()

            # Алгоритм уменьшения PAPR
            for j in range(N_iter):
                S_t_canc_tmp = np.zeros(Nfft, dtype=complex)

                for p in range(M_iter):
                    Nintervals = Nfft // N_samples

                    # Позиции и амплитуды пиков в интервале
                    sinc_Ampl = np.zeros(Nintervals, dtype=complex)
                    sinc_shift = np.zeros(Nintervals)
                    for k in range(Nintervals):
                        signal = S_t[k * N_samples:(k + 1) * N_samples]

                        Max_value = np.max(np.abs(signal))
                        Indx = np.argmax(np.abs(signal))

                        # Проверка превышения порога и установка параметров пика
                        if Max_value > TH_abs[j]:
                            sinc_Ampl[k] = signal[Indx] * (1 - TH_abs[j] / Max_value)
                            sinc_shift[k] = Indx + k * N_samples
                        else:
                            sinc_Ampl[k] = 0
                            sinc_shift[k] = 1

                        # Суммирование SINC для уменьшения найденных пиков
                        S_t_canc_tmp += sinc_Ampl[k] * np.roll(SINC_t, int(sinc_shift[k] - 1))

                        # Суммирование дельта-функций для уменьшения найденных пиков
                        S_t_ant_canc_peak[i1, i2, int(Indx + k * N_samples)] += sinc_Ampl[k]

                    S_t -= S_t_canc_tmp
                    S_t_canc[i1, i2, :] += S_t_canc_tmp

            # Модифицированный сигнал после уменьшения PAPR
            S_t_ant_new[i1, i2, :] = S_t

    # Проверка в домене ЦАП
    S_t_dac_canc_sig = np.zeros((Nsym, Ndac, Nfft), dtype=complex)
    S_t_dac_canc_peak = np.zeros((Nsym, Ndac, Nfft), dtype=complex)
    S_t_ant_canc_peak_new = np.zeros((Nsym, Nant, Nfft), dtype=complex)

    for i1 in range(Nsym):  # Для каждого символа
        for i3 in range(Nfft):  # Для каждого временного сэмпла
            sig = S_t_ant_canc_peak[i1, :, i3]

            # Проверка наличия найденных пиков
            indx = np.where(np.abs(sig) > 0)[0]

            if indx.size > 0:
                A_new = A[indx, :]
                sig_new = sig[indx]
                s1 = np.linalg.pinv(A_new) @ sig_new  # Трансформация только пиков
                s2 = np.linalg.pinv(A) @ sig  # Трансформация пиков + нулей
                S_t_dac_canc_peak[i1, :, i3] = coef1 * s1 + coef2 * s2
            else:
                S_t_dac_canc_peak[i1, :, i3] = np.zeros(Ndac)

        # Возвращение в домен антенн
        S_t_ant_canc_peak_new[i1, :, :] = A @ S_t_dac_canc_peak[i1, :, :]

        # Конволюция пиков с функциями SINC в домене ЦАП для соответствия маске спектра
        for i2 in range(Ndac):  # Для каждого ЦАП
            s_t = S_t_dac_canc_peak[i1, i2, :]
            S_t_dac_canc_sig[i1, i2, :] = s_t @ SINC_mtx

    # Проверка EVM и пропорциональное уменьшение шума CFR
    EVM_approx = np.sqrt(np.sum(np.abs(S_t_dac_canc_sig) ** 2) / np.sum(np.abs(S_t_ant) ** 2))
    S_t_dac_canc_sig /= max(EVM_approx / max_EVM, 1)

    # Генерация итогового сигнала в антеннах для анализа PAPR
    S_t_ant_new2 = np.zeros((Nsym, Nant, Nfft), dtype=complex)
    for i1 in range(Nsym):
        S_t_ant_new2[i1, :, :] = A @ S_t_dac_canc_sig[i1, :, :]

    # Возвращение сигнала в изначальный формат (символы, отсчеты, антенны)
    dS = S_t_ant_new2
    dX = np.transpose(dS, (0, 2, 1))
    return dX