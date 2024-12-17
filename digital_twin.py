import numpy as np

def digital_twin(S, Fa, Fd, scenario):
    Ntx = scenario['Ntx']
    Nsc = scenario['Nsc']
    Nsym = scenario['Nsym']
    
    # Initialize the output array X with zeros
    X = np.zeros((Nsym, Nsc, Ntx), dtype=complex)
    
    # Perform the nested loop computation
    for sym in range(Nsym):
        for sc in range(Nsc):
            # Perform the matrix multiplications
            scaled_result = Fa @ (Fd[sc, :, :] @ S[sym, sc, :])
            X[sym, sc, :] = scaled_result
    
    return X

    # X = np.zeros((Nsym, Nsc, Ntx))

    # Fa = np.array(Fa)

    # for sym in range(Nsym):
    #     for sc in range(Nsc):
    #         intermediate = np.dot(Fd[sc, :, :], S[sym, sc, :])  # Shape: (16,)

    #         # If Fa is (64, 16), apply element-wise scaling
    #         if Fa.shape == (Ntx, intermediate.shape[0]):
    #             scaled_result = Fa @ intermediate  # Shape: (64,)
    #         else:
    #             raise ValueError(f"Incompatible Fa shape: {Fa.shape}")

    #         # Assign result to X
    #         X[sym, sc, :] = scaled_result


    # return X