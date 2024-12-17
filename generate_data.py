import numpy as np

def generate_data(scenario):
    Nsym = scenario["Nsym"]
    Nue = scenario["Nue"]
    Nsc = scenario["Nsc"]

    


    QAM_order = scenario["QAM_order"]
    QAM_points = 2 ** QAM_order

    data_bits = np.random.randint(0, 2, (QAM_order, Nsym * Nsc * Nue))  # QAM_order - n of rows, Nsym*Nsc*Nue - n of cols

    data_syms = bit2int(data_bits, QAM_order)


    data_syms = qammod(data_syms, QAM_points)  # Gray coding, phase offset = 0

    X = data_syms.reshape((Nsym, Nsc, Nue))

    return X

def bit2int(bits, order):
    weights = (2 ** np.arange(order - 1, -1, -1)).reshape(order, 1)
    
    # Compute the integer values
    num = np.sum(bits * weights, axis=0)
    return num

def qammod(data, M, unit_avg_power=True):

    # Check if M is a power of 2
    if not (np.log2(M)).is_integer():
        raise ValueError("M must be a power of 2.")
    
    # Define QAM constellation
    n = int(np.sqrt(M))  # n x n square constellation (for M=16, n=4)
    
    # Generate constellation points
    # Create grid of real and imaginary parts
    x = np.arange(-n + 1, n, 2)  # Odd spaced points
    constellation = np.array([complex(re, im) for im in x[::-1] for re in x])
    
    # Map input data symbols to constellation points
    modulated_data = constellation[data]
    
    # Normalize to unit average power if required
    if unit_avg_power:
        avg_power = np.mean(np.abs(modulated_data)**2)
        modulated_data = modulated_data / np.sqrt(avg_power)  # Normalize to unit average power
    
    return modulated_data

