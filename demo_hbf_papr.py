import numpy as np
import matplotlib.pyplot as plt
from tester import tester
from time import time
from concurrent.futures import ProcessPoolExecutor


def scenario_worker(index, scenario, ML_coef):
    np.random.seed(index + 1)
    scen = scenario.copy()
    return tester(scen, ML_coef)


def scenarios_validation_parallel(scenario, ML_coef, num_workers=16):

    N_scen = scenario['N_scen']
    metrics = [{'CCDF': None} for _ in range(N_scen)]
    indices = range(N_scen)
    
    # Parallel execution using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(scenario_worker, indices, [scenario] * N_scen, [ML_coef] * N_scen))
    
    for index_i, result in enumerate(results):
        metrics[index_i] = result

    return metrics

def scenarios_validation_for_numba(scenario, ML_coef, num_workers=8):
    N_scen = scenario['N_scen']
    metrics = [{'CCDF': None} for _ in range(N_scen)]
    indices = np.arange(N_scen)  # Используем массив индексов

    # Результаты инициализируем как массив объектов
    results = np.empty(N_scen, dtype=np.object_)
    
    for i in range(len(indices)):  # Цикл с параллельной обработкой
        index = indices[i]
        results[index] = scenario_worker(index, scenario, ML_coef)

    
    # Параллельная обработка
 

    # Конвертируем результаты обратно в список для согласованности с оригинальной функцией
    for index_i, result in enumerate(results):
        metrics[index_i] = result

    return metrics


if __name__ == "__main__":
    np.random.seed(42)

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
        'PAPR_algo': 1,
        'modulation_method': 1
    }


    PAPR_thresholds = [8, 8, 8]  # Порог PAPR в dB


    # scenario['PAPR_algo'] = 3

    # N_values = [20]  # Number of scenarios
    # execution_times = []



    # for N in N_values:
    #     scenario['N_scen'] = N
    #     start_time = time()
    #     metrics = scenarios_validation_parallel(scenario, PAPR_thresholds, num_workers=8)
    #     end_time = time()
    #     execution_times.append(end_time - start_time)


    # # Plot execution time
    # plt.figure()
    # plt.plot(N_values, execution_times, marker='o', label="8 processes")
    # plt.xlabel("Number of Scenarios (N)")
    # plt.ylabel("Execution Time (seconds)")
    # plt.title("Execution Time vs Number of Scenarios ")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # print(execution_times)


    # plt.figure()
    # plt.plot(N_values, execution_times, marker='o', label="Execution Time")
    # plt.xlabel("Number of Scenarios (N)")
    # plt.ylabel("Execution Time (seconds)")
    # plt.title("Execution Time vs Number of Scenarios")
    # plt.grid(True)
    # plt.legend()
    # plt.show()







    scenario['N_scen'] = 15

    modulation_methods = {
    1: "QAM",
    # 2: "Spiral",
#    3: "Custom Method"
}   
    legend_labels = []

    for modulation in modulation_methods:
        scenario['modulation_method'] = modulation
        for papr_algo in [1, 3]:  # 1, 3, 
            scenario['PAPR_algo'] = papr_algo

            # Запуск симуляции
            start_time = time()
            if papr_algo == 5:
                 metrics = scenarios_validation_for_numba(scenario, PAPR_thresholds)
            else:
                 metrics = scenarios_validation_parallel(scenario, PAPR_thresholds)
            
            end_time = time()

            # Лог времени выполнения
            print(f'Modulation: {modulation_methods[modulation]}, '
                f'PAPR_algo: {papr_algo}, '
                f'Time in sec: {end_time - start_time}')

            # Получение данных CCDF
            ccdf_data = np.mean(np.array([m['CCDF'] for m in metrics]), axis=0)

            # Построение графика
            plt.semilogy(scenario['papr_ccdf'], ccdf_data)

            # Добавление подписи в легенду
            legend_labels.append(f"{modulation_methods[modulation]} (PAPR_algo {papr_algo})")

    # Оформление графика
    plt.legend(legend_labels)
    plt.xlabel("PAPR Threshold (dB)")
    plt.ylabel("CCDF")
    plt.title("Comparison of Modulation Methods and PAPR Reduction Algorithms")
    plt.grid(True, which='both')
    plt.show()