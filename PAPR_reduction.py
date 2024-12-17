import numpy as np
from ft_transform import ft_transform
# from SINC_2D import SINC_2D
from SINC_paper import sinc_paper


def PAPR_reduction(X, Fa, scenario, ML_coef):
    max_evm = scenario['max_evm']
    
    Ntx = scenario['Ntx']
    Nsc = scenario['Nsc']
    Nfft = scenario['Nfft']
    Nsym = scenario['Nsym']
    
    X_t = ft_transform(X, "f2t", scenario)
    
    def db2mag(db):
        db = np.asarray(db)
        return 10 ** (db / 20)

    threshold = db2mag(ML_coef)  # dB
    algo = scenario['PAPR_algo']
    
    if algo == 1:  # For original signal calculation
        dX_t = np.zeros((Nsym, Nfft, Ntx))
    # elif algo == 2:
    #     dX_t = SINC_2D(X_t, Fa, Nsc, threshold, max_evm, Nfft, scenario)
    elif algo == 3:
        dX_t = sinc_paper(X_t, Fa, Nsc, threshold, max_evm, Nfft, scenario)  # SINC_paper
    
    dX = ft_transform(dX_t, "t2f", scenario)
    
    Xm = np.zeros((Nsym, Nsc, Ntx), dtype=complex)  # Ensure Xm is complex
    for sym in range(Nsym):
        dZ = np.squeeze(dX[sym, :, :]) @ np.linalg.pinv(Fa).T  # Put it into STR to avoid this code
        dX_feasible = dZ @ Fa.T
        
        Xm[sym, :, :] = np.squeeze(X[sym, :, :]) - dX_feasible  # Complex subtraction
    
    return Xm