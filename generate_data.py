import numpy as np

def generate_data(scenario):
    Nsym = scenario["Nsym"]
    Nue = scenario["Nue"]
    Nsc = scenario["Nsc"]
    modulation = scenario['modulation_method'] 


    QAM_order = scenario["QAM_order"]
    QAM_points = 2 ** QAM_order

    data_bits = np.random.randint(0, 2, (QAM_order, Nsym * Nsc * Nue))  # QAM_order - n of rows, Nsym*Nsc*Nue - n of cols

    data_syms = bit2int(data_bits, QAM_order)

    if modulation == 1:
        data_syms = qammod(data_syms, QAM_points)  # Gray coding, phase offset = 0
    elif modulation == 2:
        data_syms = circularmod(data_syms, QAM_points) 
    elif modulation == 3:
        data_syms = clipped_qammod(data_syms, QAM_points) 
        

    X = data_syms.reshape((Nsym, Nsc, Nue))

    return X

def bit2int(bits, order):
    weights = (2 ** np.arange(order - 1, -1, -1)).reshape(order, 1) #like 0 1 1 0 -> 2^1 + 2^2 = 6
    num = np.sum(bits * weights, axis=0)
    return num


def circularmod(data, M, alpha=1.0, beta=np.pi / 4, unit_avg_power=True):
    """
    Модуляция сигнала с использованием спирали вместо квадратной сетки.
    
    :param data: Индексы символов, которые нужно модулировать.
    :param M: Число точек созвездия (должно быть >= len(data)).
    :param alpha: Коэффициент увеличения радиуса спирали.
    :param beta: Угол между соседними точками (в радианах).
    :param unit_avg_power: Нормализация к средней мощности 1.
    :return: Комплексные точки созвездия.
    """
    # Создание спиральной созвездии
    indices = np.arange(M)  # Индексы точек
    radii = alpha * indices  # Радиус зависит линейно от индекса
    angles = beta * indices  # Угол между соседними точками
    
    # Комплексные точки созвездия
    constellation = radii * np.exp(1j * angles)
    
    # Сопоставление входных данных точкам созвездия
    modulated_data = constellation[data]
    
    # Нормализация мощности (если включено)
    if unit_avg_power:
        avg_power = np.mean(np.abs(modulated_data) ** 2)
        modulated_data = modulated_data / np.sqrt(avg_power)
    
    return modulated_data



def qammod(data, M, unit_avg_power=True):

    # Check if M is a power of 2
    if not (np.log2(M)).is_integer():
        raise ValueError("M must be a power of 2.")
    
    # Define QAM constellation
    n = int(np.sqrt(M))  # n x n square constellation 
    
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


def clipped_qammod(data, M, max_amplitude=2.0, unit_avg_power=True):
    """
    QAM с ограничением максимальной амплитуды для снижения PAPR.
    
    :param data: Индексы символов.
    :param M: Размер созвездия.
    :param max_amplitude: Максимальная амплитуда точек.
    :param unit_avg_power: Нормализация к средней мощности 1.
    :return: Модулированные точки.
    """
    # Создание классического QAM созвездия
    n = int(np.sqrt(M))
    x = np.arange(-n + 1, n, 2)
    constellation = np.array([complex(re, im) for im in x[::-1] for re in x])

    # Ограничение амплитуды
    amplitudes = np.abs(constellation)
    constellation = np.where(amplitudes > max_amplitude, 
                              max_amplitude * constellation / amplitudes, 
                              constellation)
    
    # Сопоставление данных
    modulated_data = constellation[data]

    # Нормализация мощности
    if unit_avg_power:
        avg_power = np.mean(np.abs(modulated_data) ** 2)
        modulated_data = modulated_data / np.sqrt(avg_power)

    return modulated_data
