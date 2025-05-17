import numpy as np
import neurokit2 as nk
import pandas as pd

def procesar_archivo(archivo=None):
    sampling_rate = 256 
    duration = 10 
    eeg_signal = nk.eeg_simulate(duration=duration,
                                sampling_rate=sampling_rate,
                                noise=0.05)
    time = np.linspace(0, duration, sampling_rate * duration)
    df = pd.DataFrame({"Time": time, "EEG": eeg_signal})
    return df

def run_model(data):
    print("Ejecutando modelo...")
    # Simulando un modelo de predicción
    # TODO Aquí deberías incluir la lógica de tu modelo

    simulated_model = np.random.rand(6)
    simulated_model = simulated_model / np.sum(simulated_model)
    print("Modelo simulado:", simulated_model)
    print("Predicción:", sum(simulated_model))
    return simulated_model
