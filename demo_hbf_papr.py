import numpy as np
import matplotlib.pyplot as plt
from tester import tester

np.random.seed(42)

def scenarios_validation(scenario, ML_coef):
    N_scen = scenario['N_scen']
    metrics = [{'CCDF': None} for _ in range(N_scen)]

    for index_i in range(N_scen):
        np.random.seed(index_i + 1)

        scen = scenario.copy()
        scen['index'] = index_i + (1 if index_i > 23 else 0)  # 24 is very bad scenario, something wrong with precoding?

        metrics[index_i] = tester(scen, ML_coef)

    return metrics

scenario = {
    'Ntx': 64,  # transmit antennas
    'Ndac': 16,  # digital-to-analog converters
    'Nue': 16,  # single-antenna users
    'Nsc': 192,
    'Nsym': 2,  # symbols in time
    'Nfft': 2048,
    'QAM_order': 8,
    'max_evm': 0.125,
    'papr_ccdf': np.arange(5, 15.1, 0.1),
    'N_scen': 12,  # Number of scenarios to simulate
    'PAPR_algo': 1
}


PAPR_thresholds = [8, 8, 8]  # Порог PAPR в dB


plt.figure()
metrics = scenarios_validation(scenario, PAPR_thresholds)


ccdf_data = np.mean(np.array([m['CCDF'] for m in metrics]), axis=0)
plt.semilogy(scenario['papr_ccdf'], ccdf_data)


scenario['PAPR_algo'] = 3
metrics = scenarios_validation(scenario, PAPR_thresholds)
ccdf_data = np.mean(np.array([m['CCDF'] for m in metrics]), axis=0)
plt.semilogy(scenario['papr_ccdf'], ccdf_data)


plt.legend(["Original signal", "PAPR reduction"])
plt.xlabel("PAPR Threshold (dB)")
plt.ylabel("CCDF")
plt.grid(True, which='both')
plt.show()