import numpy as np

def ft_transform(X, mode, scenario):
    # X shape: (Nsym, Nsc, Ntx)
    
    Ntx = scenario['Ntx']
    Nfft = scenario['Nfft']
    Nsym = scenario['Nsym']
    Nsc = scenario['Nsc']
    
    shift = Nsc // 2
    
    if mode == "f2t":
        X_f = np.zeros((Nsym, Nfft, Ntx), dtype=X.dtype)
        X_f[:, :Nsc, :] = X
        X_f = np.roll(X_f, -shift, axis=1)
        
        Xtransformed = np.fft.ifft(X_f, Nfft, axis=1) * np.sqrt(Nfft)
    elif mode == "t2f":
        X_f = np.fft.fft(X, Nfft, axis=1) / np.sqrt(Nfft)
        X_f = np.roll(X_f, shift, axis=1)
        Xtransformed = X_f[:, :Nsc, :]
    
    return Xtransformed

