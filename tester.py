import numpy as np
from generate_data import generate_data
from analog_beamforming import analog_beamforming
from digital_twin import digital_twin
from PAPR_reduction import PAPR_reduction
from compute_metrics import compute_metrics


def tester(scenario, ML_coef):
    Nsc = scenario["Nsc"]
    Nue = scenario["Nue"]
    Ntx = scenario["Ntx"]

    # Generate data for transmission
    S = generate_data(scenario)

    # Load channel (ideal CE assumed)
    H = (np.random.randn(Nsc, Nue, Ntx) + 1j * np.random.randn(Nsc, Nue, Ntx)) / np.sqrt(2)

    # Compute analog beamforming (beamspace selection)
    Fa = analog_beamforming(H, scenario)

    # Compute digital beamforming (no precoding - eye)
    Fd = np.tile(np.reshape(np.eye(Nue), (1, Nue, Nue)), (Nsc, 1, 1))

    # Compute TX signals before PAPR reduction
    X = digital_twin(S, Fa, Fd, scenario)

    # PAPR compensated signal
    Xm = PAPR_reduction(X, Fa, scenario, ML_coef)

    # Compute PAPR, CCDF
    metrics = compute_metrics(Xm, scenario)
    
    return metrics

